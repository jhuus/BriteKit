import csv
import importlib
import threading

import pytest
import requests
from click.testing import CliRunner

from britekit.commands import inat

module = importlib.import_module("britekit.commands._inat")
AUDIO = b"ID3\x04\x00\x00\x00\x00\x00\x00audio payload"


class Response:
    def __init__(self, content=AUDIO, status=200):
        self.content = content
        self.status_code = status

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        pass

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError("HTTP error")


def observation(index, *, quality="research", sounds=None):
    return {
        "id": index + 1000,
        "quality_grade": quality,
        "sounds": (
            sounds
            if sounds is not None
            else [{"file_url": f"https://example.com/{index}.m4a?token=value"}]
        ),
    }


@pytest.fixture
def fake_http(monkeypatch):
    state = {
        "observations": [observation(i) for i in range(6)],
        "responses": {},
        "calls": [],
        "sessions": [],
        "metadata": [],
    }

    def get_observations(**params):
        state["metadata"].append(params)
        return {"results": state["observations"]}

    monkeypatch.setattr("pyinaturalist.get_observations", get_observations)

    class Session:
        def __init__(self):
            self.closed = False
            state["sessions"].append(self)

        def get(self, url, **kwargs):
            assert kwargs == {"allow_redirects": True, "timeout": (10, 60)}
            state["calls"].append((self, url))
            response = state["responses"].get(url, Response())
            if isinstance(response, Exception):
                raise response
            return response

        def close(self):
            self.closed = True

    monkeypatch.setattr(module, "create_download_session", Session)
    return state


def read_map(path):
    with (path / "inat.csv").open(newline="") as f:
        return {r["Media ID"]: r["URL"] for r in csv.DictReader(f)}


def test_parallel_downloads_reuse_per_worker_sessions(tmp_path, fake_http, monkeypatch):
    barrier = threading.Barrier(2)
    create_session = module.create_download_session

    def make_session():
        session = create_session()
        get = session.get

        def synchronized_get(url, **kwargs):
            barrier.wait(timeout=5)
            return get(url, **kwargs)

        session.get = synchronized_get
        return session

    monkeypatch.setattr(module, "create_download_session", make_session)
    inat(name="Bird", output_dir=str(tmp_path), max_downloads=4, workers=2)
    assert len(list(tmp_path.glob("N*.m4a"))) == 4
    assert len(fake_http["sessions"]) == 2
    assert all(s.closed for s in fake_http["sessions"])
    assert set(read_map(tmp_path)) == {"0", "1", "2", "3"}
    assert fake_http["metadata"] == [
        {"taxon_name": "Bird", "sounds": True, "page": "all", "per_page": 200}
    ]


def test_limit_counts_existing_converted_audio_and_preserves_mapping(
    tmp_path, fake_http
):
    (tmp_path / "N0.mp3").write_bytes(AUDIO)
    (tmp_path / "inat.csv").write_text(
        "Media ID,URL\nold,https://www.inaturalist.org/observations/9\n"
    )
    inat(name="Bird", output_dir=str(tmp_path), max_downloads=3, workers=4)
    assert sorted(p.name for p in tmp_path.glob("N*")) == ["N0.mp3", "N1.m4a", "N2.m4a"]
    assert len(fake_http["calls"]) == 2
    assert read_map(tmp_path) == {
        "old": "https://www.inaturalist.org/observations/9",
        "0": "https://www.inaturalist.org/observations/1000",
        "1": "https://www.inaturalist.org/observations/1001",
        "2": "https://www.inaturalist.org/observations/1002",
    }
    inat(name="Bird", output_dir=str(tmp_path), max_downloads=3)
    assert len(fake_http["calls"]) == 2


def test_invalid_downloads_do_not_consume_limit_or_corrupt_files(tmp_path, fake_http):
    (tmp_path / "N3.m4a").write_bytes(b"<html>old failed download</html>")
    responses = [
        Response(status=503),
        Response(content=b"<html>bad</html>"),
        requests.Timeout(),
    ]
    for i, response in enumerate(responses):
        fake_http["responses"][f"https://example.com/{i}.m4a?token=value"] = response
    inat(name="Bird", output_dir=str(tmp_path), max_downloads=2, workers=4)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "N3.m4a",
        "N4.m4a",
        "inat.csv",
    ]
    assert (tmp_path / "N3.m4a").read_bytes() == AUDIO
    assert set(read_map(tmp_path)) == {"3", "4"}


def test_multiple_sounds_duplicate_urls_and_verification_filters(tmp_path, fake_http):
    fake_http["observations"] = [
        observation(0, quality="needs_id"),
        observation(1, quality="casual", sounds=[{"file_url": None}, {"file_url": ""}]),
        observation(
            2,
            sounds=[{"file_url": f"https://example.com/{i}.mp3"} for i in range(1, 5)],
        ),
        observation(
            3, sounds=[{"file_url": "https://example.com/1.mp3?duplicate=yes"}]
        ),
    ]
    inat(name="Bird", output_dir=str(tmp_path), max_downloads=2)
    assert sorted(p.name for p in tmp_path.glob("N*")) == ["N1.mp3", "N2.mp3"]
    assert len(fake_http["calls"]) == 2
    assert read_map(tmp_path)["1"] == "https://www.inaturalist.org/observations/1002"


def test_no_prefix_unverified_and_unlimited(tmp_path, fake_http):
    fake_http["observations"] = [
        observation(0, quality="needs_id"),
        observation(1, quality="casual"),
    ]
    inat(
        name="Bird",
        output_dir=str(tmp_path),
        max_downloads=0,
        no_prefix=True,
        include_unverified=True,
    )
    assert sorted(p.name for p in tmp_path.iterdir()) == ["0.m4a", "1.m4a", "inat.csv"]
    assert len(fake_http["calls"]) == 2


def test_atomic_write_failure_keeps_existing_file_and_csv(
    tmp_path, fake_http, monkeypatch
):
    target = tmp_path / "N0.m4a"
    target.write_bytes(b"original invalid audio")
    previous_csv = "Media ID,URL\nold,https://example.com/old\n"
    (tmp_path / "inat.csv").write_text(previous_csv)
    fake_http["observations"] = [observation(0)]

    def replace(*_args):
        raise OSError("disk failure")

    monkeypatch.setattr(module.os, "replace", replace)
    with pytest.raises(OSError, match="disk failure"):
        inat(name="Bird", output_dir=str(tmp_path))
    assert target.read_bytes() == b"original invalid audio"
    assert (tmp_path / "inat.csv").read_text() == previous_csv
    assert list(tmp_path.glob("*.part")) == []


def test_empty_results_do_not_create_mapping(tmp_path, fake_http):
    fake_http["observations"] = []
    inat(name="Bird", output_dir=str(tmp_path))
    assert list(tmp_path.iterdir()) == []
    assert fake_http["sessions"] == []


def test_cli_worker_option_and_validation(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(module, "inat", lambda *args: calls.append(args))
    runner = CliRunner()
    args = ["--name", "Bird", "--output", str(tmp_path)]
    assert runner.invoke(module._inat_cmd, args + ["--workers", "2"]).exit_code == 0
    assert calls[0] == ("Bird", str(tmp_path), 500, False, False, 2)
    assert runner.invoke(module._inat_cmd, args + ["--workers", "0"]).exit_code == 2
