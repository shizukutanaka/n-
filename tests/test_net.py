"""Local-endpoint HTTP helpers must ignore env proxy variables.

All nmesh services bind to loopback; a corporate HTTP_PROXY/HTTPS_PROXY
env can never serve 127.0.0.1 but urllib/httpx honor it by default and
would route health probes and upstream calls through a dead proxy.
"""

from __future__ import annotations

import queue as queue_mod
import socket
import threading
import urllib.request

from nmesh.net import local_async_client, local_client, local_urlopen


def _serve_once() -> tuple[int, queue_mod.Queue[bytes]]:
    """One-shot HTTP server capturing the request line."""
    requests: queue_mod.Queue[bytes] = queue_mod.Queue()
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]

    def run() -> None:
        conn, _ = listener.accept()
        requests.put(conn.recv(4096).split(b"\r\n", 1)[0])
        conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n")
        conn.close()
        listener.close()

    threading.Thread(target=run, daemon=True).start()
    return port, requests


def test_local_urlopen_bypasses_env_proxy(monkeypatch) -> None:
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")
    port, requests = _serve_once()
    with local_urlopen(f"http://127.0.0.1:{port}/", timeout=2) as response:
        assert response.status == 200
    # An origin-form request line proves the request was not proxied.
    assert requests.get(timeout=5).startswith(b"GET /")


def test_env_proxy_would_misroute_loopback(monkeypatch) -> None:
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")
    try:
        urllib.request.urlopen("http://127.0.0.1:1/", timeout=2)
    except OSError:
        pass
    else:  # pragma: no cover - depends on platform proxy resolution
        raise AssertionError("env proxy unexpectedly bypassed")


def test_local_client_disables_env_trust() -> None:
    assert local_client().trust_env is False
    assert local_async_client().trust_env is False
