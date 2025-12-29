#!/usr/bin/env python3

import warnings


def pytest_configure():
    warnings.filterwarnings("ignore", message=".*TorchCodec.*")
    warnings.filterwarnings("ignore", message=".*StreamingMediaDecoder.*")
