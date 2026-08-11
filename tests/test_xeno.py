import json
import logging

from britekit.commands import xeno


class _Response:
    def __init__(self, *, content=b"", data=None):
        self.status_code = 200
        self.content = content
        self.text = json.dumps(data) if data is not None else ""


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
            _Response(content=b"one"),
            _Response(content=b"two"),
        ]
    )
    monkeypatch.setattr("requests.get", lambda *_args, **_kwargs: next(responses))

    with caplog.at_level(logging.INFO):
        xeno(key="key", name="Bird", output_dir=str(tmp_path), max_downloads=2)

    assert "Downloaded 2 recordings" in caplog.messages
    assert sorted(path.name for path in tmp_path.iterdir()) == ["XC0.mp3", "XC1.mp3"]


def test_xeno_reports_zero_when_files_already_exist(tmp_path, monkeypatch, caplog):
    (tmp_path / "XC1.mp3").write_bytes(b"existing")
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
        "requests.get", lambda *_args, **_kwargs: _Response(data=api_response)
    )

    with caplog.at_level(logging.INFO):
        xeno(key="key", name="Bird", output_dir=str(tmp_path))

    assert "Downloaded 0 recordings" in caplog.messages
