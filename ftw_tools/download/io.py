import shutil
import tempfile
from pathlib import Path

import fsspec

DEFAULT_HEADERS = {"User-Agent": "Wget/1.21.4"}


def copy_url_to_file(url: str, destination: str | Path) -> Path:
    """Copy a remote URL to a local file via fsspec."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f"{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as target:
        with fsspec.open(url, "rb", headers=DEFAULT_HEADERS) as source:
            shutil.copyfileobj(source, target)

    Path(target.name).replace(destination)
    return destination
