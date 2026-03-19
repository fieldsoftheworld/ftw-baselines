import os
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen


DEFAULT_USER_AGENT = "Wget/1.21.4"
CHUNK_SIZE = 1024 * 1024


def download_url_to_path(
    url: str,
    destination: str | Path,
    *,
    user_agent: str = DEFAULT_USER_AGENT,
    progress_cb=None,
) -> Path:
    """Download a URL to a local path with an explicit user agent.

    Some remote hosts reject urllib's default Python user agent with HTTP 403.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f"{destination.name}.",
        suffix=".tmp",
    )
    os.close(fd)
    Path(temp_name).unlink(missing_ok=True)
    temp_path = Path(temp_name)

    request = Request(url, headers={"User-Agent": user_agent})

    try:
        with urlopen(request) as response, open(temp_path, "wb") as output:
            total = int(response.headers.get("Content-Length", 0))
            current = 0

            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                output.write(chunk)
                current += len(chunk)
                if progress_cb is not None:
                    progress_cb(current, total)

        temp_path.replace(destination)
        return destination
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
