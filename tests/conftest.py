import os
import sys
from pathlib import Path
from unittest.mock import patch

import matplotlib
import pytest

# Force matplotlib to use a non-interactive backend and disable showing plots
os.environ.setdefault("MPLBACKEND", "Agg")
matplotlib.use("Agg", force=True)
matplotlib.interactive(False)


@pytest.fixture(scope="session", autouse=True)
def mock_crop_calendar_downloads(request):
    # don't mock for integration tests
    if request.node.get_closest_marker("integration"):
        yield
        return

    with patch(
        "ftw_tools.download.crop_calendar.get_crop_calendar_cache_dir",
        return_value=Path(__file__).parent / "data-files" / "crop-calendar",
    ):
        yield


def pytest_sessionfinish(session, exitstatus):
    """Attempt to clean up background workers that may keep pytest alive. Only applies to macosx silicon with python 3.12."""
    if (
        sys.platform == "darwin"
        and sys.version_info.major == 3
        and sys.version_info.minor == 12
    ):
        os._exit(0)
