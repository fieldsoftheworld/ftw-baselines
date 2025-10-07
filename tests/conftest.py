import os
import sys

import matplotlib

# Force matplotlib to use a non-interactive backend and disable showing plots
os.environ.setdefault("MPLBACKEND", "Agg")
matplotlib.use("Agg", force=True)
matplotlib.interactive(False)


def pytest_sessionfinish(session, exitstatus):
    """Attempt to clean up background workers that may keep pytest alive. Only applies to macosx silicon with python 3.12."""
    if (
        sys.platform == "darwin"
        and sys.version_info.major == 3
        and sys.version_info.minor == 12
    ):
        os._exit(0)
