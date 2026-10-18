import os
import socket
import sys

import matplotlib
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("YOLO_OFFLINE", "true")
matplotlib.use("Agg", force=True)
matplotlib.interactive(False)

LOCAL_HOSTS = {None, "localhost", "127.0.0.1", "::1"}


@pytest.fixture(autouse=True)
def block_network(request, monkeypatch):
    """Fail tests that reach the network unless they are marked integration."""
    if request.node.get_closest_marker("integration"):
        return
    connect, getaddrinfo = socket.socket.connect, socket.getaddrinfo

    def blocked(host):
        raise OSError(f"Network access to {host} in a test not marked integration")

    def guarded_connect(sock, address):
        if isinstance(address, tuple) and address[0] not in LOCAL_HOSTS:
            blocked(address[0])
        return connect(sock, address)

    def guarded_getaddrinfo(host, *args, **kwargs):
        if host not in LOCAL_HOSTS:
            blocked(host)
        return getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)
    monkeypatch.setenv("GDAL_HTTP_PROXY", "http://127.0.0.1:9")


def pytest_sessionfinish(session, exitstatus):
    """Attempt to clean up background workers that may keep pytest alive. Only applies to macosx silicon with python >= 3.12."""
    if (
        not hasattr(session.config, "workerinput")
        and sys.platform == "darwin"
        and sys.version_info.major == 3
        and sys.version_info.minor >= 12
    ):
        os._exit(0)
