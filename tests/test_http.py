from pathlib import Path
from unittest.mock import patch

from ftw_tools.download.http import DEFAULT_USER_AGENT, download_url_to_path


class _FakeResponse:
    def __init__(self, body: bytes):
        self._body = body
        self.headers = {"Content-Length": str(len(body))}

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = len(self._body)
        chunk = self._body[:size]
        self._body = self._body[size:]
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_download_url_to_path_sets_user_agent(tmp_path: Path):
    body = b"ftw"
    destination = tmp_path / "file.bin"

    with patch("ftw_tools.download.http.urlopen") as mock_urlopen:
        mock_urlopen.return_value = _FakeResponse(body)

        download_url_to_path("https://example.com/file.bin", destination)

    request = mock_urlopen.call_args.args[0]
    assert request.headers["User-agent"] == DEFAULT_USER_AGENT
    assert destination.read_bytes() == body
