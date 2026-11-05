"""HTTP helpers for downloading audio from public recording archives."""


def create_download_session():
    """Reuse connections and back off on transient server/network failures."""
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        respect_retry_after_header=True,
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.mount("http://", HTTPAdapter(max_retries=retry))
    return session


def is_audio_header(content: bytes) -> bool:
    """Reject empty/error pages while accepting common audio containers."""
    return (
        content.startswith((b"ID3", b"OggS", b"fLaC"))
        or (len(content) >= 2 and content[0] == 0xFF and content[1] & 0xE0 == 0xE0)
        or (content.startswith(b"RIFF") and content[8:12] == b"WAVE")
        or content[4:8] == b"ftyp"
    )
