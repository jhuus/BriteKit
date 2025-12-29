#!/usr/bin/env python3


class DatabaseError(Exception):
    """Raised for database-specific errors."""

    pass


class InferenceError(Exception):
    """Raised for inference-specific errors."""

    pass


class ModelError(Exception):
    """Raised for errors related to models."""

    pass


class TrainingError(Exception):
    """Raised for training-specific errors."""

    pass


class InputError(Exception):
    """Raised for other input errors."""

    pass
