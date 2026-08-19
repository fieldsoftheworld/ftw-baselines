import os
import sys

import matplotlib

os.environ.setdefault("MPLBACKEND", "Agg")
matplotlib.use("Agg", force=True)
matplotlib.interactive(False)


def pytest_sessionfinish(session, exitstatus):
    """Attempt to clean up background workers that may keep pytest alive. Only applies to macosx silicon with python >= 3.12."""
    if (
        not hasattr(session.config, "workerinput")
        and sys.platform == "darwin"
        and sys.version_info.major == 3
        and sys.version_info.minor >= 12
    ):
        os._exit(0)
