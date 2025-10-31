# Defer some imports to improve initialization performance.
import logging
import pickle
from typing import Any, List, Tuple

from pytorch_lightning import LightningDataModule

from britekit.core.config_loader import get_config


class DataModule(LightningDataModule):
    def __init__(self):
        super().__init__()
        from britekit.core.dataset import SpectrogramDataset

        self.cfg = get_config()
        self.train_data = None
        self.val_data = None
        self.test_data = None

        # Load training data
        try:
            class_names, class_codes, alt_names, alt_codes, specs, labels = (
                self._load_pickle_data(self.cfg.train.train_pickle)
            )

            # Validate loaded data
            if not class_names or not specs or not labels:
                raise ValueError("Training data is empty or invalid")

            if len(specs) != len(labels):
                raise ValueError(
                    f"Mismatch between specs ({len(specs)}) and labels ({len(labels)}) lengths"
                )

            self.train_class_names = class_names
            self.train_class_codes = class_codes
            self.train_class_alt_names = alt_names
            self.train_class_alt_codes = alt_codes
            self.specs = specs
            self.labels = labels
            self.num_train_classes = len(class_names)

            # flatten the labels so [[1, 2], [3]] becomes [1, 2, 3]
            self.flattened_labels = [
                item for sublist in self.labels for item in sublist
            ]

            self.num_train_specs = len(self.specs)
            logging.info(
                f"Fetched {self.num_train_specs} spectrograms for {len(self.train_class_names)} classes"
            )
        except Exception as e:
            logging.error(f"Failed to load training data: {e}")
            raise

        if (
            self.cfg.train.noise_class_name
            and self.cfg.train.noise_class_name in self.train_class_names
        ):
            noise_class_index = self.train_class_names.index(
                self.cfg.train.noise_class_name
            )
        else:
            noise_class_index = -1

        self.full_dataset = SpectrogramDataset(
            self.specs,
            self.labels,
            len(self.train_class_names),
            noise_class_index,
            is_training=True,
        )

        # Load test data
        if self.cfg.train.test_pickle:
            try:
                class_names, class_codes, alt_names, alt_codes, specs, labels = (
                    self._load_pickle_data(self.cfg.train.test_pickle)
                )

                # Validate test data
                if not class_names or not specs or not labels:
                    logging.error(
                        "Test data is empty or invalid, setting test_data to None"
                    )
                    self.test_data = None
                elif len(specs) != len(labels):
                    logging.error(
                        f"Mismatch between test specs ({len(specs)}) and labels ({len(labels)}) lengths, setting test_data to None"
                    )
                    self.test_data = None
                else:
                    self.test_specs = specs
                    self.test_labels = labels

                    self.test_data = SpectrogramDataset(
                        self.test_specs,
                        self.test_labels,
                        len(class_names),
                        is_training=False,
                    )
            except Exception as e:
                logging.error(
                    f"Failed to load test data: {e}, setting test_data to None"
                )
                self.test_data = None
        else:
            self.test_data = None

        if self.cfg.train.num_folds > 1:
            # Stratified k-fold split
            from sklearn.model_selection import StratifiedKFold

            skf = StratifiedKFold(
                n_splits=self.cfg.train.num_folds, shuffle=True, random_state=42
            )
            self.indices = list(skf.split(self.specs, self.labels))

    def _load_pickle_data(
        self, path: str
    ) -> Tuple[List[str], List[str], List[str], List[str], List[Any], List[List[int]]]:
        """
        Load data from a pickle file with error handling.

        Args:
        - path (str): Path to the pickle file

        Returns:
            Tuple containing (class_names, class_codes, alt_names, alt_codes, specs, labels)

        Raises:
            FileNotFoundError: If the pickle file doesn't exist
            ValueError: If the pickle file is corrupted or missing required keys
        """
        if not path:
            raise ValueError("Pickle file path cannot be empty")

        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Pickle file not found: {path}")
        except (pickle.UnpicklingError, EOFError) as e:
            raise ValueError(f"Failed to load pickle file {path}: {e}")
        except Exception as e:
            raise ValueError(f"Unexpected error loading pickle file {path}: {e}")

        # Validate required keys exist
        required_keys = [
            "class_names",
            "class_codes",
            "alt_names",
            "alt_codes",
            "spec_values",
            "spec_class_indexes",
        ]
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            raise ValueError(
                f"Pickle file {path} missing required keys: {missing_keys}"
            )

        return (
            data["class_names"],
            data["class_codes"],
            data["alt_names"],
            data["alt_codes"],
            data["spec_values"],
            data["spec_class_indexes"],
        )

    def prepare_fold(self, fold_index: int):
        """
        Prepare train/validation split for a specific fold.

        Args:
        - fold_index (int): Index of the fold to prepare

        Raises:
            ValueError: If fold_index is invalid or val_portion is invalid
        """
        from torch.utils.data import Subset

        if not hasattr(self, "full_dataset") or self.full_dataset is None:
            raise ValueError("Full dataset not initialized")

        if self.cfg.train.num_folds <= 1:
            # Simple train/val split
            if not (0 <= self.cfg.train.val_portion < 1):
                raise ValueError(
                    f"val_portion must be between 0 and 1, got {self.cfg.train.val_portion}"
                )

            val_size = int(len(self.full_dataset) * self.cfg.train.val_portion)
            train_size = len(self.full_dataset) - val_size

            if train_size <= 0:
                raise ValueError(
                    f"Invalid split sizes: train_size={train_size}, val_size={val_size}"
                )

            indices = list(range(len(self.full_dataset)))
            train_indices = indices[:train_size]
            val_indices = indices[train_size:]

            self.train_data = Subset(self.full_dataset, train_indices)
            self.val_data = Subset(self.full_dataset, val_indices)
        else:
            # Stratified k-fold split
            if not hasattr(self, "indices") or not self.indices:
                raise ValueError("K-fold indices not initialized")

            if fold_index < 0 or fold_index >= len(self.indices):
                raise ValueError(
                    f"fold_index {fold_index} out of range [0, {len(self.indices)})"
                )

            train_idx, val_idx = self.indices[fold_index]
            self.train_data = Subset(self.full_dataset, train_idx)
            self.val_data = Subset(self.full_dataset, val_idx)

    def train_dataloader(self):
        from torch.utils.data import DataLoader

        if self.train_data is None:
            raise ValueError("Training data not prepared. Call prepare_fold() first.")
        return DataLoader(
            self.train_data,
            batch_size=self.cfg.train.batch_size,
            shuffle=self.cfg.train.shuffle,
            num_workers=self.cfg.train.num_workers,
        )

    def val_dataloader(self):
        from torch.utils.data import DataLoader

        if self.val_data is None:
            raise ValueError("Validation data not prepared. Call prepare_fold() first.")
        return DataLoader(
            self.val_data,
            batch_size=self.cfg.train.batch_size,
            shuffle=False,
            num_workers=self.cfg.train.num_workers,
        )

    def test_dataloader(self):
        from torch.utils.data import DataLoader

        if self.test_data is None:
            logging.error("Test data not available, returning None")
            return None
        return DataLoader(
            self.test_data,
            batch_size=self.cfg.train.batch_size,
            shuffle=False,
            num_workers=self.cfg.train.num_workers,
        )
