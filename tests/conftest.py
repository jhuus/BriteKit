# tests/conftest.py
import warnings

def pytest_configure():
    warnings.filterwarnings(
        "ignore",
        message=".*TorchCodec.*"
    )
    warnings.filterwarnings(
        "ignore",
        message=".*StreamingMediaDecoder.*"
    )
