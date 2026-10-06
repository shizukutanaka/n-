from __future__ import annotations

import asyncio
import http.server
import threading
import urllib.request

import httpx
import pytest

from nmesh.net import local_async_client, local_client, local_urlopen

_DEAD_PROXY = "http://127.0.0.1:9"


@pytest.fixture()
def loopback_server():
    server = http.server.HTTPServer(
        ("127.0.0.1", 0), http.server.SimpleHTTPRequestHandler
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    thread.join(timeout=5)


@pytest.fixture()
def poisoned_proxy_env(monkeypatch):
    for name in (
        "http_proxy", "https_proxy", "all_proxy",
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
    ):
        monkeypatch.setenv(name, _DEAD_PROXY)
    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.delenv("NO_PROXY", raising=False)


def test_local_urlopen_bypasses_env_proxy(loopback_server, poisoned_proxy_env):
    with pytest.raises(OSError):
        urllib.request.urlopen(f"{loopback_server}/", timeout=2)
    with local_urlopen(f"{loopback_server}/", timeout=2) as response:
        assert response.status == 200


def test_local_client_bypasses_env_proxy(loopback_server, poisoned_proxy_env):
    with pytest.raises(httpx.TransportError):
        httpx.Client().get(f"{loopback_server}/", timeout=2.0)
    with local_client() as client:
        response = client.get(f"{loopback_server}/", timeout=2.0)
    assert response.status_code == 200


def test_local_async_client_bypasses_env_proxy(loopback_server, poisoned_proxy_env):
    async def fetch() -> int:
        async with local_async_client() as client:
            response = await client.get(f"{loopback_server}/", timeout=2.0)
        return response.status_code

    assert asyncio.run(fetch()) == 200


def test_supervisor_health_probe_bypasses_env_proxy(
    loopback_server, poisoned_proxy_env
):
    """Regression: health probes must reach loopback even with a dead proxy
    configured — previously a proxied probe made live engines look dead."""
    from nmesh.runtime.supervisor import Supervisor

    assert Supervisor._health_url_alive(f"{loopback_server}/")


def test_bounded_get_aborts_on_oversized_body():
    from nmesh.net import _MAX_BODY_BYTES, bounded_get

    oversized = _MAX_BODY_BYTES + 1

    def handler(_request) -> httpx.Response:
        return httpx.Response(200, content=b"x" * oversized)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(ValueError, match="body limit"):
        bounded_get(client, "http://example/feed")


def test_bounded_get_passes_small_body_through():
    from nmesh.net import bounded_get

    def handler(_request) -> httpx.Response:
        return httpx.Response(200, content=b'{"ok": true}')

    client = httpx.Client(transport=httpx.MockTransport(handler))
    response = bounded_get(client, "http://example/feed")
    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_bounded_read_aborts_on_oversized_body():
    from nmesh.net import _MAX_BODY_BYTES, bounded_read

    class _Response:
        def __init__(self, data: bytes) -> None:
            self._data = data

        def read(self, n: int = -1) -> bytes:
            return self._data[:n]

    with pytest.raises(ValueError, match="body limit"):
        bounded_read(_Response(b"x" * (_MAX_BODY_BYTES + 1)))  # type: ignore[arg-type]


def test_bounded_read_returns_small_body():
    from nmesh.net import bounded_read

    class _Response:
        def __init__(self, data: bytes) -> None:
            self._data = data

        def read(self, n: int = -1) -> bytes:
            return self._data[:n]

    assert bounded_read(_Response(b"payload")) == b"payload"  # type: ignore[arg-type]


def test_bounded_get_decodes_compressed_body_once():
    import gzip

    from nmesh.net import bounded_get

    def handler(_request) -> httpx.Response:
        return httpx.Response(
            200,
            content=gzip.compress(b'{"ok": true}'),
            headers={"Content-Encoding": "gzip"},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    response = bounded_get(client, "http://example/feed")
    assert response.json() == {"ok": True}
