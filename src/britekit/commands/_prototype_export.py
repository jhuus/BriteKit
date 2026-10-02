"""Export class-specific prototypes and their strongest training examples."""

import heapq
import json
from pathlib import Path
import pickle
from typing import Optional

import click


def prototype_export(
    train_pickle_path: str,
    checkpoint_path: str,
    output_path: str,
    top_k: int = 5,
    batch_size: int = 16,
    device: Optional[str] = None,
) -> None:
    """Save prototype vectors, readout weights and top matching examples.

    Search all supplied examples, without augmentation or a same-species filter,
    so spurious matches remain visible. The pickle must use the checkpoint's
    frontend and class ordering. Activation locations are feature-map cells,
    not exact event boundaries or receptive-field bounds.
    """
    import numpy as np
    import torch

    from britekit.core import util
    from britekit.core.dataset import SpectrogramDataset
    from britekit.models import model_loader
    from britekit.models.prototype_head import PrototypeSEDHead

    if top_k < 1 or batch_size < 1:
        raise ValueError("top_k and batch_size must be positive")
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(f"Output directory already exists: {destination}")
    with open(train_pickle_path, "rb") as file:
        data = pickle.load(file)
    required = (
        "class_codes",
        "class_names",
        "spec_values",
        "spec_class_indexes",
        "spec_segment_ids",
        "spec_recording_ids",
    )
    if any(key not in data for key in required):
        raise ValueError(f"Training pickle must contain {required}")
    count = len(data["spec_values"])
    if count == 0 or any(len(data[key]) != count for key in required[3:]):
        raise ValueError("Training pickle has empty or inconsistent sample arrays")
    if len(set(data["spec_segment_ids"])) != count:
        raise ValueError("Training pickle contains duplicate segment IDs")
    model = model_loader.load_from_checkpoint(checkpoint_path).eval()
    head = model.head
    if not isinstance(head, PrototypeSEDHead) or model.backbone is None:
        raise ValueError("Checkpoint must have a prototype_sed head")
    if (
        list(data["class_codes"]) != model.train_class_codes
        or list(data["class_names"]) != model.train_class_names
    ):
        raise ValueError("Training pickle must match checkpoint species ordering")
    device = device or util.get_device()
    model.to(device)
    dataset = SpectrogramDataset(
        data["spec_values"],
        data["spec_class_indexes"],
        model.num_classes,
        is_training=False,
    )
    # One heap per prototype, with one best location per input example.
    heaps: list[list[tuple[float, int, int, int]]] = [
        [] for _ in range(model.num_classes * head.prototypes_per_class)
    ]
    with torch.inference_mode():
        for start in range(0, count, batch_size):
            batch = torch.stack(
                [
                    dataset[i]["input"]
                    for i in range(start, min(start + batch_size, count))
                ]
            ).to(device)
            maps = head.similarity_maps(model.backbone(batch)).flatten(1, 2)
            scores, locations = maps.flatten(2).max(dim=-1)
            if not torch.isfinite(scores).all():
                raise ValueError("Non-finite prototype similarities")
            width = maps.shape[-1]
            scores, locations = scores.cpu().tolist(), locations.cpu().tolist()
            for b, row in enumerate(scores):
                for p, score in enumerate(row):
                    entry = (
                        score,
                        start + b,
                        locations[b][p] // width,
                        locations[b][p] % width,
                    )
                    if len(heaps[p]) < top_k:
                        heapq.heappush(heaps[p], entry)
                    elif entry > heaps[p][0]:
                        heapq.heapreplace(heaps[p], entry)

        destination.mkdir(parents=True)
        np.savez_compressed(
            destination / "prototypes.npz",
            vectors=head.prototypes.detach().cpu().numpy(),
            weights=head.weights.detach().cpu().numpy(),
            bias=head.bias.detach().cpu().numpy(),
            class_codes=np.asarray(model.train_class_codes),
            class_names=np.asarray(model.train_class_names),
        )
        records = []
        selected: dict[int, set[int]] = {}
        for p, heap in enumerate(heaps):
            c, j = divmod(p, head.prototypes_per_class)
            matches = []
            for score, index, frequency, time in sorted(heap, reverse=True):
                selected.setdefault(index, set()).add(p)
                matches.append(
                    dict(
                        segment_id=int(data["spec_segment_ids"][index]),
                        recording_id=int(data["spec_recording_ids"][index]),
                        sample_index=index,
                        similarity=score,
                        frequency_cell=frequency,
                        time_cell=time,
                        labeled_class_codes=[
                            model.train_class_codes[k]
                            for k in data["spec_class_indexes"][index]
                        ],
                        example_file=f"example-{index}.npz",
                    )
                )
            records.append(
                dict(
                    class_index=c,
                    class_code=model.train_class_codes[c],
                    class_name=model.train_class_names[c],
                    prototype_index=j,
                    matches=matches,
                )
            )
        # Save each retained input once, along with only its selected maps.
        for index, prototype_ids in sorted(selected.items()):
            spec = dataset[index]["input"]
            maps = head.similarity_maps(
                model.backbone(spec.unsqueeze(0).to(device))
            ).flatten(1, 2)[0]
            ids = sorted(prototype_ids)
            np.savez_compressed(
                destination / f"example-{index}.npz",
                spectrogram=spec.numpy(),
                prototype_ids=np.asarray(ids),
                similarity_maps=maps[ids].cpu().numpy(),
            )
        manifest = dict(
            checkpoint_path=str(Path(checkpoint_path).resolve()),
            training_pickle=str(Path(train_pickle_path).resolve()),
            prototypes_per_class=head.prototypes_per_class,
            top_k=top_k,
            examples_searched=count,
            audio_config=util.cfg_to_pure(model.cfg.audio),
            location_units="native feature-map cells; not exact sound boundaries",
            prototypes=records,
        )
        (destination / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )


@click.command(
    name="prototype-export",
    help="Export class-specific prototypes and their top matching training examples.",
)
@click.option(
    "--train-pickle",
    "train_pickle_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--checkpoint",
    "checkpoint_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
)
@click.option("--output", "output_path", required=True, type=click.Path())
@click.option("--top-k", default=5, type=click.IntRange(min=1), show_default=True)
@click.option("--batch-size", default=16, type=click.IntRange(min=1), show_default=True)
@click.option("--device", default=None)
def _prototype_export_cmd(**kwargs):
    prototype_export(**kwargs)
