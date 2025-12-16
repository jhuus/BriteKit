import torch
import torch.nn as nn


class ModelInspector:
    """
    Hooks into any PyTorch model and prints output shapes for each module.
    Also produces a compact stage-level summary.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.handles: list = []
        self.layer_info: list = []  # (name, class, in_shape, out_shape)

    def _hook(self, name):
        def fn(module, inp, out):
            in_shape = (
                tuple(inp[0].shape)
                if isinstance(inp, (list, tuple))
                else tuple(inp.shape)
            )
            if isinstance(out, (list, tuple)):
                out_shape = [tuple(o.shape) for o in out]
            else:
                out_shape = tuple(out.shape)

            self.layer_info.append(
                (name, module.__class__.__name__, in_shape, out_shape)
            )

        return fn

    def register_hooks(self):
        """
        Attach a forward hook to every leaf module.
        """

        for name, module in self.model.named_modules():
            # Skip containers
            if isinstance(module, (nn.Sequential, nn.ModuleList, nn.Identity)):
                continue
            # Only hook modules that change shapes or compute
            if any(
                isinstance(module, t)
                for t in [
                    nn.Conv2d,
                    nn.BatchNorm2d,
                    nn.Linear,
                    nn.ReLU,
                    nn.SiLU,
                    nn.AvgPool2d,
                    nn.AdaptiveAvgPool2d,
                    nn.MaxPool2d,
                ]
            ):
                self.handles.append(module.register_forward_hook(self._hook(name)))

    def remove(self):
        for h in self.handles:
            h.remove()

    def run(self, x: torch.Tensor, verbose: bool = True):
        """
        Pass an input through the model and capture all shapes.
        Returns list of (name, class, in_shape, out_shape).
        """
        print(f"ModelInspector.run start {x.shape=}")
        self.register_hooks()
        _ = self.model(x)
        self.remove()

        if verbose:
            self.print_report()

        return self.layer_info
        print("ModelInspector.run end")

    def print_report(self, file=None):
        def write(line):
            if file:
                file.write(line + "\n")
            else:
                print(line)

        prev_stage = None
        for name, cls, in_s, out_s in self.layer_info:
            if "stage" in name and prev_stage != name.split(".")[0]:
                prev_stage = name.split(".")[0]
                write(f"\n-- {prev_stage} ----------------------------")
            write(f"{name:40s} │ {cls:18s} │ {in_s} -> {out_s}")
