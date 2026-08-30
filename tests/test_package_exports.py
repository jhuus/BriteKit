import subprocess
import sys


def test_package_import_does_not_eagerly_load_optional_stacks():
    script = """
import sys
import britekit
from britekit import Audio

excluded_prefixes = (
    "britekit.commands",
    "britekit.core.trainer",
    "britekit.core.tuner",
    "britekit.testing",
    "britekit.training_db",
)
assert not any(name.startswith(excluded_prefixes) for name in sys.modules)
assert Audio.__module__ == "britekit.core.audio"
"""

    subprocess.run([sys.executable, "-c", script], check=True)


def test_public_exports_are_loaded_lazily_and_cached():
    import britekit
    from britekit.core.audio import Audio

    assert britekit.Audio is Audio
    assert britekit.Audio is britekit.Audio
    assert "Audio" in britekit.__dict__


def test_public_api_names_remain_listed():
    import britekit

    assert "Audio" in britekit.__all__
    assert "Predictor" in dir(britekit)
    assert "OccurrencePickleProvider" in dir(britekit)
