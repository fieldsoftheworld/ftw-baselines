import os
import shutil
import tempfile
from pathlib import Path

import fsspec

DEFAULT_HEADERS = {"User-Agent": "Wget/1.21.4"}


def copy_url_to_file(url: str, destination: str | Path, *, headers=None) -> Path:
    """Copy a remote URL to a local file via fsspec."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    headers = DEFAULT_HEADERS if headers is None else headers

    fd, temp_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f"{destination.name}.",
        suffix=".tmp",
    )
    os.close(fd)
    temp_path = Path(temp_name)

    try:
        with fsspec.open(url, "rb", headers=headers) as source, open(
            temp_path, "wb"
        ) as target:
            shutil.copyfileobj(source, target)

        temp_path.replace(destination)
        return destination
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
