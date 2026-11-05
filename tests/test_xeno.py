import json
import logging
import importlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import requests

from britekit.commands import xeno


class _Response:
    def __init__(self, *, content=b"ID3audio", data=None, status=200):
        self.status_code = status
        self.content = content
        self.text = json.dumps(data) if data is not None else ""

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        pass

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError("HTTP error")


def test_xeno_reports_number_downloaded_at_limit(tmp_path, monkeypatch, caplog):
    api_response = {
        "page": 1,
        "numPages": 1,
        "recordings": [
            {
                "id": str(index),
                "q": "A",
                "lic": "by-nc",
                "animal-seen": "yes",
                "file": f"https://example.com/{index}.mp3",
                "url": "",
            }
            for index in range(3)
        ],
    }
    responses = iter(
        [
            _Response(data=api_response),
            _Response(content=b"ID3one"),
            _Response(content=b"ID3two"),
        ]
    )
    monkeypatch.setattr(
        "requests.Session.get", lambda *_args, **_kwargs: next(responses)
    )

    with caplog.at_level(logging.INFO):
        xeno(
            key="key", name="Bird", output_dir=str(tmp_path), max_downloads=2, workers=1
        )

    assert "Downloaded 2 recordings" in caplog.messages
    assert sorted(path.name for path in tmp_path.iterdir()) == ["XC0.mp3", "XC1.mp3"]


def test_xeno_reports_zero_when_files_already_exist(tmp_path, monkeypatch, caplog):
    (tmp_path / "XC1.mp3").write_bytes(b"ID3existing")
    api_response = {
        "page": 1,
        "numPages": 1,
        "recordings": [
            {
                "id": "1",
                "q": "A",
                "lic": "by-nc",
                "animal-seen": "yes",
                "file": "https://example.com/1.mp3",
                "url": "",
            }
        ],
    }
    monkeypatch.setattr(
        "requests.Session.get", lambda *_args, **_kwargs: _Response(data=api_response)
    )

    with caplog.at_level(logging.INFO):
        xeno(key="key", name="Bird", output_dir=str(tmp_path))

    assert "Downloaded 0 recordings" in caplog.messages


def _recording(index):
    return {
        "id": str(index),
        "q": "A",
        "lic": "by-nc",
        "animal-seen": "yes",
        "file": f"https://example.com/{index}.mp3",
        "url": "",
    }


def test_concurrent_limit_includes_existing_and_replaces_bad_files(
    tmp_path, monkeypatch, caplog
):
    (tmp_path / "XC99.mp3").write_bytes(b"ID3existing")
    (tmp_path / "XC3.mp3").write_bytes(b"<html>old error</html>")
    audio_calls = []
    recordings = [_recording(0)] + [_recording(i) for i in range(8)]

    def get(_session, url, **kwargs):
        assert kwargs["timeout"] == (10, 60)
        if "/api/3/" in url:
            return _Response(
                data={"page": "1", "numPages": "1", "recordings": recordings}
            )
        audio_calls.append(url)
        if url.endswith("0.mp3"):
            return _Response(status=403)
        if url.endswith("1.mp3"):
            return _Response(content=b"<html>Rate limited</html>")
        if url.endswith("2.mp3"):
            return _Response(content=b"")
        return _Response()

    monkeypatch.setattr("requests.Session.get", get)
    with caplog.at_level(logging.INFO):
        xeno(key="secret-key", name="Bird", output_dir=str(tmp_path), max_downloads=3)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "XC3.mp3",
        "XC4.mp3",
        "XC99.mp3",
    ]
    assert (tmp_path / "XC3.mp3").read_bytes() == b"ID3audio"
    assert len(audio_calls) == len(set(audio_calls)) == 5
    assert "secret-key" not in caplog.text


def test_limit_already_met_skips_metadata(tmp_path, monkeypatch):
    (tmp_path / "XC1.mp3").write_bytes(b"ID3existing")

    def get(*_args, **_kwargs):
        raise AssertionError("Should not request metadata when limit is met")

    monkeypatch.setattr("requests.Session.get", get)
    xeno(key="key", name="Bird", output_dir=str(tmp_path), max_downloads=1)


def test_sessions_reused_and_downloads_overlap(tmp_path, monkeypatch):
    barrier = threading.Barrier(2)
    sessions = []
    recordings = [_recording(i) for i in range(4)]

    def get(session, url, **_kwargs):
        if "/api/3/" in url:
            return _Response(data={"page": 1, "numPages": 1, "recordings": recordings})
        sessions.append(session)
        barrier.wait(timeout=5)
        return _Response()

    monkeypatch.setattr("requests.Session.get", get)
    xeno(key="key", name="Bird", output_dir=str(tmp_path), max_downloads=4, workers=2)
    assert len(list(tmp_path.iterdir())) == 4
    assert len({id(s) for s in sessions}) == 2


def test_empty_search_returns_without_pagination_loop(tmp_path, monkeypatch):
    calls = []

    def get(*_args, **_kwargs):
        calls.append(True)
        return _Response(data={"page": "1", "numPages": "0", "recordings": []})

    monkeypatch.setattr("requests.Session.get", get)
    xeno(key="key", name="Bird", output_dir=str(tmp_path))
    assert len(calls) == 1


def test_http_session_retries_429_and_503():
    module = importlib.import_module("britekit.commands._xeno")
    statuses = [429, 503, 200]
    requests_seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests_seen.append(self.path)
            status = statuses.pop(0)
            self.send_response(status)
            self.send_header("Retry-After", "0")
            self.end_headers()
            self.wfile.write(b"ID3audio" if status == 200 else b"temporary error")

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with module._make_session() as session:
            session.get_adapter("http://").max_retries.backoff_factor = 0
            with session.get(
                f"http://127.0.0.1:{server.server_port}/audio", timeout=(1, 2)
            ) as response:
                response.raise_for_status()
                assert response.content == b"ID3audio"
            assert session.get_adapter("http://").max_retries.respect_retry_after_header
        assert requests_seen == ["/audio"] * 3
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_paginated_quality_selection_preserves_filters(tmp_path, monkeypatch):
    recordings = [_recording(i) for i in range(5)]
    recordings[0]["q"] = "C"
    recordings[2]["lic"] = "by-nc-nd"
    recordings[3]["animal-seen"] = "no"
    pages = iter([recordings[:3], recordings[3:]])
    downloaded = []

    def get(_session, url, **_kwargs):
        if "/api/3/" in url:
            page = 1 if "page=1&" in url else 2
            return _Response(
                data={"page": page, "numPages": 2, "recordings": next(pages)}
            )
        downloaded.append(url)
        return _Response()

    monkeypatch.setattr("requests.Session.get", get)
    xeno(
        key="key",
        name="Bird",
        output_dir=str(tmp_path),
        max_downloads=2,
        workers=1,
        seen_only=True,
    )
    assert downloaded == ["https://example.com/1.mp3", "https://example.com/4.mp3"]


def test_unlimited_downloads_exhaust_candidates(tmp_path, monkeypatch):
    def get(_session, url, **_kwargs):
        if "/api/3/" in url:
            return _Response(
                data={
                    "page": 1,
                    "numPages": 1,
                    "recordings": [_recording(i) for i in range(5)],
                }
            )
        return _Response()

    monkeypatch.setattr("requests.Session.get", get)
    xeno(key="key", name="Bird", output_dir=str(tmp_path), max_downloads=0, workers=2)
    assert len(list(tmp_path.iterdir())) == 5
