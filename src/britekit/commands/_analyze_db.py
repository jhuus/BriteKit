# File name starts with _ to keep it out of typeahead for API users.
# Defer some imports to improve --help performance.
import logging
import os
from pathlib import Path
from typing import Optional

import click

from britekit.core.config_loader import get_config
from britekit.core.util import cli_help_from_doc, set_logging


class DatabaseAnalyzer:
    def __init__(
        self,
        cfg_path: Optional[str] = None,
        db_path: Optional[str] = None,
        class_name: Optional[str] = None,
        classes_path: Optional[str] = None,
        spec_group: str = "default",
        output_path: str = "",
        plot: bool = False,
        max_score: float = 0.95,
    ):
        from britekit.core.predictor import Predictor

        self.cfg = get_config(cfg_path)
        self.db_path = self.cfg.train.train_db if db_path is None else db_path
        self.class_name = class_name
        self.classes_path = classes_path
        self.spec_group = spec_group
        self.output_path = output_path
        self.plot = plot
        self.max_score = max_score

        self.predictor = Predictor(self.cfg.misc.ckpt_folder)
        assert self.predictor.class_names is not None
        assert self.predictor.class_codes is not None

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def analyze(self):
        import numpy as np
        import pandas as pd
        from britekit.training_db.training_db import TrainingDatabase

        class_index_dict = {}
        for i, name in enumerate(self.predictor.class_names):
            class_index_dict[name] = i

        # create list of class names
        if self.class_name is None:
            if self.classes_path is None:
                class_names = self.predictor.class_names
            else:
                df = pd.read_csv(self.classes_path)
                if "Name" in df:
                    class_names = df["Name"].astype(str).tolist()
                    for name in class_names:
                        if name not in class_index_dict:
                            logging.error(
                                f"Error: class '{name}' not found in trained models."
                            )
                            return
                else:
                    logging.error(
                        f"Error: 'Name' column not found in {self.classes_path}."
                    )
                    return
        else:
            if self.class_name not in class_index_dict:
                logging.error(
                    f"Error: class '{self.class_name}' not found in trained models."
                )
                return

            class_names = [self.class_name]

        # init columns for class summary
        names = []
        codes = []
        counts = []
        quantile_10s = []
        quantile_20s = []
        quantile_30s = []
        means = []
        fp1_names = []
        fp1_means = []
        fp2_names = []
        fp2_means = []
        fp3_names = []
        fp3_means = []

        # loop on classes
        with TrainingDatabase(self.db_path) as db:
            for name in class_names:
                results = db.get_class({"name": name})
                if len(results) == 0:
                    logging.error(
                        f"Error. Class {name} not found in database {self.db_path}."
                    )
                    return

                result = results[0]
                names.append(name)
                codes.append(result.code)

                specs = db.get_spectrogram_by_class(name, spec_group=self.spec_group)
                logging.info(f"Analyzing {len(specs)} spectrograms for {name}")
                counts.append(len(specs))

                scores = self._get_scores(specs)
                index = class_index_dict[name]

                self._save_class_details(index, scores, specs)
                if self.plot:
                    self._plot_specs(name, result.code, scores[:, index], specs)

                # get basic stats for this class
                quantile_10s.append(np.quantile(scores[:, index], 0.10))
                quantile_20s.append(np.quantile(scores[:, index], 0.20))
                quantile_30s.append(np.quantile(scores[:, index], 0.30))
                class_means = scores.mean(axis=0)
                means.append(class_means[index])

                # find up to three classes with the next highest mean scores
                N = min(3, len(self.predictor.class_names) - 1)
                masked_means = class_means.copy()
                masked_means[index] = -np.inf  # exclude current class
                indices = np.argpartition(masked_means, -N)[-N:]
                indices = indices[
                    np.argsort(masked_means[indices])[::-1]
                ]  # sort descending
                topN = [(i, class_means[i]) for i in indices]
                fp1_names.append(self.predictor.class_names[topN[0][0]])
                fp1_means.append(topN[0][1])
                if N >= 2:
                    fp2_names.append(self.predictor.class_names[topN[1][0]])
                    fp2_means.append(topN[1][1])
                else:
                    fp2_names.append("")
                    fp2_means.append("")
                if N >= 3:
                    fp3_names.append(self.predictor.class_names[topN[2][0]])
                    fp3_means.append(topN[2][1])
                else:
                    fp3_names.append("")
                    fp3_means.append("")

        df = pd.DataFrame()
        df["Name"] = names
        df["Code"] = codes
        df["Spectrograms"] = counts
        df["Mean"] = means
        df["Quantile .1"] = quantile_10s
        df["Quantile .2"] = quantile_20s
        df["Quantile .3"] = quantile_30s
        df["FP1 Name"] = fp1_names
        df["FP1 Mean"] = fp1_means
        df["FP2 Name"] = fp2_names
        df["FP2 Mean"] = fp2_means
        df["FP3 Name"] = fp3_names
        df["FP3 Mean"] = fp3_means

        df.to_csv(
            os.path.join(self.output_path, "class_summary.csv"),
            index=False,
            float_format="%.3f",
        )

        # create overall summary report
        mean_of_means = np.array(means).mean()
        mean_of_q10s = np.array(quantile_10s).mean()
        mean_of_q20s = np.array(quantile_20s).mean()
        mean_of_q30s = np.array(quantile_30s).mean()

        out_lines = []
        out_lines.append(f"Mean of score means = {mean_of_means:.3f}")
        out_lines.append(f"Mean of score quantile 0.1 = {mean_of_q10s:.3f}")
        out_lines.append(f"Mean of score quantile 0.2 = {mean_of_q20s:.3f}")
        out_lines.append(f"Mean of score quantile 0.3 = {mean_of_q30s:.3f}")

        with open(os.path.join(self.output_path, "summary.txt"), "w") as out_file:
            for out_line in out_lines:
                logging.info(out_line)
                out_file.write(out_line + "\n")

        logging.info(f"See output reports in {self.output_path}")

    def _get_scores(self, specs):
        """Run inference on a block of spectrograms."""
        import numpy as np
        from britekit.core.util import expand_spectrogram

        spec_array = np.zeros(
            (len(specs), 1, self.cfg.audio.spec_height, self.cfg.audio.spec_width),
            dtype=np.float32,
        )

        for i, spec in enumerate(specs):
            spec = expand_spectrogram(spec.value)
            spec = spec.reshape(
                (1, self.cfg.audio.spec_height, self.cfg.audio.spec_width)
            )

            spec_array[i] = spec

        scores, _ = self.predictor.get_block_scores(spec_array)
        return scores

    def _plot_specs(self, class_name, class_code, scores, specs):
        """Plot spectrograms for a class in ascending order of score."""
        import numpy as np
        from britekit.core.plot import plot_spec
        from britekit.core.util import expand_spectrogram

        # create output directory
        if class_code:
            spec_dir = os.path.join(self.output_path, class_code)
        else:
            spec_dir = os.path.join(self.output_path, class_name)

        if not os.path.exists(spec_dir):
            os.makedirs(spec_dir)

        # plot them
        logging.info("Plotting spectrograms")
        sorted_indexes = np.argsort(scores)
        for i, index in enumerate(sorted_indexes):
            spec, score = specs[index], scores[index]
            if score > self.max_score:
                break

            spec_value = expand_spectrogram(spec.value)
            stem = Path(spec.filename).stem
            image_name = f"{i+1}-{stem}-{spec.offset:.2f}-{score:.3f}.jpeg"
            plot_spec(spec_value, os.path.join(spec_dir, image_name))

    def _save_class_details(self, class_index, scores, specs):
        """Save a details CSV for the given class."""
        import numpy as np
        import pandas as pd

        numbers = []
        segment_ids = []
        filenames = []
        offsets = []
        saved_scores = []  # "scores" is the input array
        fp_names = []
        fp_codes = []
        fp_scores = []

        class_scores = scores[:, class_index]
        sorted_indexes = np.argsort(class_scores)
        for i, index in enumerate(sorted_indexes):
            spec, score = specs[index], class_scores[index]
            if score > self.max_score:
                break

            numbers.append(i + 1)
            segment_ids.append(spec.segment_id)
            filenames.append(spec.filename)
            offsets.append(spec.offset)
            saved_scores.append(score)

            # find the other class with the highest score
            segment_scores = scores[index].copy()
            segment_scores[class_index] = -np.inf
            top_indices = np.argsort(segment_scores)
            fp1_class_index = top_indices[-1]
            fp_names.append(self.predictor.class_names[fp1_class_index])
            fp_codes.append(self.predictor.class_codes[fp1_class_index])
            fp_scores.append(segment_scores[fp1_class_index])

        if len(filenames) > 0:
            df = pd.DataFrame()
            df["#"] = numbers
            df["Segment ID"] = segment_ids
            df["Recording"] = filenames
            df["Offset"] = offsets
            df["Score"] = saved_scores
            df["FP Name"] = fp_names
            df["FP Code"] = fp_codes
            df["FP Score"] = fp_scores

            class_name = self.predictor.class_names[class_index]
            class_code = self.predictor.class_codes[class_index]
            if class_code:
                csv_path = os.path.join(self.output_path, f"{class_code}.csv")
            else:
                csv_path = os.path.join(self.output_path, f"{class_name}.csv")

            df.to_csv(csv_path, index=False, float_format="%.3f")


def analyze_db(
    cfg_path: Optional[str] = None,
    db_path: Optional[str] = None,
    class_name: Optional[str] = None,
    classes_path: Optional[str] = None,
    spec_group: str = "default",
    output_path: str = "",
    plot: bool = False,
    max_score: float = 0.95,
):
    """
    Run inference on segments in a training database.

    Running inference on a training database can be used to identify bad or difficult
    training segments, or to identify classes that are likely to be mistaken for each other.

    Args:
    - cfg_path (str): Path to YAML configuration file defining model and inference settings.
    - db_path (str): Path to database to analyze.
    - class_name (str): Optional class name. By default, do all classes.
    - classes_path (str): Optional path to CSV listing classes to process. By default, do all classes.
    - spec_group (str): Spectrogram group name. Defaults to 'default'.
    - output_path (str): Path to output directory where results will be saved.
    - plot (bool): If specified, plot spectrograms per class by ascending score.
    - max_score (float): Save details and plot only if score less than this (default = 0.95).
    """
    analyzer = DatabaseAnalyzer(
        cfg_path,
        db_path,
        class_name,
        classes_path,
        spec_group,
        output_path,
        plot,
        max_score,
    )
    analyzer.analyze()


@click.command(
    name="analyze-db",
    short_help="Run inference on a training database.",
    help=cli_help_from_doc(analyze_db.__doc__),
)
@click.option(
    "-c",
    "--cfg",
    "cfg_path",
    type=click.Path(exists=True),
    required=False,
    help="Path to YAML file defining config overrides.",
)
@click.option(
    "-d", "--db", "db_path", required=False, help="Path to the training database."
)
@click.option(
    "--name", "class_name", required=False, help="Class name. Default is all classes."
)
@click.option(
    "--classes",
    "classes_path",
    required=False,
    type=click.Path(file_okay=True, dir_okay=False),
    help="Path to CSV containing class names. Default is all classes.",
)
@click.option(
    "--sgroup",
    "spec_group",
    required=False,
    default="default",
    help="Spectrogram group name. Defaults to 'default'.",
)
@click.option(
    "-o",
    "--output",
    "output_path",
    type=click.Path(file_okay=False, dir_okay=True),
    required=True,
    help="Path to output directory.",
)
@click.option(
    "--plot",
    "plot",
    is_flag=True,
    help="If specified, plot spectrograms per class by ascending score.",
)
@click.option(
    "--max",
    "max_score",
    type=float,
    default=0.95,
    help="Save details and plot only if score less than this (default = 0.95).",
)
def _analyze_db_cmd(
    cfg_path: str,
    db_path: str,
    class_name: str,
    classes_path: str,
    spec_group: str,
    output_path: str,
    plot: bool,
    max_score: float,
):
    set_logging()

    analyze_db(
        cfg_path,
        db_path,
        class_name,
        classes_path,
        spec_group,
        output_path,
        plot,
        max_score,
    )
