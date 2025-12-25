# Defer some imports to improve initialization performance.
import logging
import os
import pickle
import random
from typing import Optional

from britekit.core.config_loader import get_config
from britekit.core.exceptions import InputError


class OccurrencePickler:
    """
    Create a pickle file from an occurrence database, for fast access during inference.

    Attributes:
        db_path (str): path to database.
        output_path (str): output_path.
    """

    def __init__(
        self,
        db_path: str,
        output_path: str,
    ):
        from britekit.occurrence_db.occurrence_db import OccurrenceDatabase

        self.cfg = get_config()
        self.output_path = output_path

        if not os.path.exists(db_path):
            raise InputError(f'Database "{db_path}" not found')

        self.db = OccurrenceDatabase(db_path)

    def _smooth(self, occurrences):
        """Smooth by setting each weekly value to the max of adjacent values."""
        import numpy as np

        smoothed = np.zeros(len(occurrences)).astype(np.float16)
        for i in range(len(occurrences)):
            smoothed[i] = (
                max(
                    max(occurrences[i], occurrences[(i + 1) % 48]),
                    occurrences[(i - 1) % 48],
                )
                .astype(np.float16)
                .item()
            )

        return smoothed

    def pickle(self, quiet=False):
        """Create the pickle file as specified."""

        import numpy as np

        # add county info
        logging.info("Fetching county info.")
        counties = self.db.get_all_counties()
        pickle_dict = {}
        pickle_dict["counties"] = {}
        for county in counties:
            pickle_dict["counties"][county.code] = county

        # add class info
        logging.info("Fetching class info.")
        classes = self.db.get_all_classes()
        pickle_dict["classes"] = {}
        for _class in classes:
            pickle_dict["classes"][_class.name] = _class

        # add occurrence info
        pickle_dict["occurrences"] = {}  # array of weekly values per county/class
        pickle_dict["smoothed"] = {}  # smoothed array of weekly values
        pickle_dict["max"] = {}  # max of weekly values

        for county in counties:
            logging.info(f"Processing occurrence data for {county.code}.")
            pickle_dict["occurrences"][county.code] = {}
            pickle_dict["smoothed"][county.code] = {}
            pickle_dict["max"][county.code] = {}

            for _class in classes:
                occurrences = self.db.get_occurrences(county.id, _class.name)
                if len(occurrences) > 0:
                    occurrences = np.array(occurrences).astype(np.float16)
                    smoothed = self._smooth(occurrences).astype(np.float16)
                    pickle_dict["occurrences"][county.code][_class.name] = occurrences
                    pickle_dict["smoothed"][county.code][_class.name] = smoothed
                    pickle_dict["max"][county.code][_class.name] = (
                        occurrences.max().astype(np.float16).item()
                    )

        # pickle it
        pickle_file = open(self.output_path, "wb")
        pickle.dump(pickle_dict, pickle_file)
        logging.info(f'Saved data in "{self.output_path}"')


class TrainingPickler:
    """
    Create a pickle file from selected training records, for input to training.

    Attributes:
        db_path (str): path to database.
        output_path (str): output_path.
        classes_path (Optional, str): path to CSV file listing classes.
        max_per_class (int, optional): maximum spectrograms to output per class.
    """

    def __init__(
        self,
        db_path: str,
        output_path: str,
        classes_path: Optional[str] = None,
        max_per_class: Optional[int] = None,
        spec_group: Optional[str] = None,
    ):
        from britekit.training_db.training_db import TrainingDatabase

        self.cfg = get_config()
        self.classes_path = classes_path
        self.output_path = output_path
        self.max_per_class = max_per_class
        self.spec_group = spec_group if spec_group is not None else "default"

        if self.max_per_class:
            # When tuning spectrogram parameters we might need to pickle multiple
            # spec groups. In that case it's important to get the same ones each
            # time, if max_per_class is specified.
            random.seed(101)

        if not os.path.exists(db_path):
            raise InputError(f'Database "{db_path}" not found')

        self.db = TrainingDatabase(db_path)

    def pickle(self, quiet=False):
        """Create the pickle file as specified."""
        import pandas as pd
        from britekit.training_db.training_data_provider import TrainingDataProvider

        # read DB and map class name to DB result
        results = self.db.get_class()
        name_dict = {}
        for r in results:
            name_dict[r.name] = r

        # get class names from CSV if specified, else from the DB
        if self.classes_path:
            df = pd.read_csv(self.classes_path)
            names = df["Name"].tolist()
        else:
            names = []
            for r in results:
                names.append(r.name)

        # get class codes, alt-names and alt-codes
        codes, alt_names, alt_codes = [], [], []
        for name in names:
            if name not in name_dict:
                raise InputError(f'Class "{name}" not found in database.')

            codes.append(name_dict[name].code)
            alt_names.append(name_dict[name].alt_name)
            alt_codes.append(name_dict[name].alt_code)

        # get dict from spec ID to class names
        segment_class_dict = TrainingDataProvider(self.db).segment_class_dict()
        name_set = set(names)

        # get spectrograms for each selected class
        total_count = 0
        empty_classes = []
        all_specs = []
        used_segment_ids = set()
        for name in names:
            specs = self.db.get_spectrogram_by_class(name, spec_group=self.spec_group)

            # segments might have multiple classes, so exclude any with unselected classes,
            # or any already used in a previous class
            valid_specs = []
            for spec in specs:
                valid_spec = True
                if spec.segment_id in used_segment_ids:
                    valid_spec = False  # used in a previous class
                else:
                    for name in segment_class_dict[spec.segment_id]:
                        if name not in name_set:
                            valid_spec = False  # matches an unselected class
                            break

                if valid_spec:
                    valid_specs.append(spec)
                    used_segment_ids.add(spec.segment_id)

            specs = valid_specs

            # pick random subset if we have more than max_per_class
            if self.max_per_class and self.max_per_class < len(specs):
                random.shuffle(specs)
                specs = specs[: self.max_per_class]

            if not quiet:
                logging.info(f'Fetched {len(specs)} spectrograms for "{name}"')

            total_count += len(specs)
            if len(specs) == 0:
                empty_classes.append(name)

            all_specs.extend(specs)

        # stop if there are classes with no spectrograms
        if len(empty_classes) > 0:
            logging.error("Error: there are no spectrograms for")
            for name in empty_classes:
                logging.info(name)

            raise InputError("Not all specified classes have spectrograms")

        if not quiet:
            logging.info(f"Total # spectrograms = {total_count}")

        # create a dict from class name to index
        class_name_dict = {}
        for i, name in enumerate(names):
            class_name_dict[name] = i

        # create lists of class indexes and values for each spectrogram
        spec_values, class_indexes = [], []
        for i, spec in enumerate(all_specs):
            spec_values.append(spec.value)
            class_indexes.append([])
            for name in segment_class_dict[spec.segment_id]:
                class_indexes[i].append(class_name_dict[name])

        # create and save the pickle file
        pickle_dict = {}
        pickle_dict["class_names"] = names
        pickle_dict["class_codes"] = codes
        pickle_dict["alt_names"] = alt_names
        pickle_dict["alt_codes"] = alt_codes
        pickle_dict["spec_values"] = spec_values
        pickle_dict["spec_class_indexes"] = class_indexes

        pickle_file = open(self.output_path, "wb")
        pickle.dump(pickle_dict, pickle_file)

        if not quiet:
            logging.info(f'Saved data in "{self.output_path}"')
