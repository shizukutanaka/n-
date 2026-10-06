"""HTTP helpers for endpoints that are always local (127.0.0.1).

All nmesh services, the gateway, and the Ollama daemon bind to loopback.
Environment proxy variables (HTTP_PROXY/HTTPS_PROXY/ALL_PROXY) — and on
macOS the system-wide proxy settings that urllib additionally consults —
can never serve a loopback address, since a proxy lives on another host.
Yet both urllib and httpx honor them by default, so on a machine
configured for a corporate proxy every health probe and upstream engine
call is silently routed through a dead end: healthy engines look dead
(restart loops), and gateway requests fail before reaching the service.
The helpers below bypass env proxies for those known-local targets;
remote endpoints (model downloads, watch sources) keep honoring the
user's proxy settings — `bounded_get`/`bounded_read` cap those remote
response bodies.
"""

from __future__ import annotations

import urllib.request
import urllib.response
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import httpx

_LOCAL_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def local_urlopen(
    request: urllib.request.Request | str, *, timeout: float | None
) -> urllib.response.addinfourl:
    """urlopen that ignores env proxy settings; for loopback endpoints."""
    return _LOCAL_OPENER.open(request, timeout=timeout)


def local_client(
    timeout: httpx.Timeout | float | None = None,
    *,
    base_url: str = "",
    follow_redirects: bool = False,
) -> httpx.Client:
    """httpx.Client that ignores env proxy settings (imported lazily)."""
    import httpx

    return httpx.Client(
        timeout=timeout,
        base_url=base_url,
        trust_env=False,
        follow_redirects=follow_redirects,
    )


def local_async_client(
    timeout: httpx.Timeout | float | None = None,
    *,
    base_url: str = "",
    follow_redirects: bool = False,
) -> httpx.AsyncClient:
    """httpx.AsyncClient that ignores env proxy settings."""
    import httpx

    return httpx.AsyncClient(
        timeout=timeout,
        base_url=base_url,
        trust_env=False,
        follow_redirects=follow_redirects,
    )


_MAX_BODY_BYTES = 16 * 1024 * 1024


def bounded_get(
    session: httpx.Client,
    url: str,
    *,
    max_bytes: int = _MAX_BODY_BYTES,
    **kwargs: Any,
) -> httpx.Response:
    """GET a remote endpoint with a capped body: remote servers can answer
    with arbitrarily large responses, so stream and abort past `max_bytes`
    instead of buffering the whole body in memory."""
    import httpx

    with session.stream("GET", url, **kwargs) as response:
        chunks: list[bytes] = []
        total = 0
        for chunk in response.iter_bytes(64 * 1024):
            total += len(chunk)
            if total > max_bytes:
                raise ValueError("upstream response exceeds the fetch body limit")
            chunks.append(chunk)
        # iter_bytes already decoded the body, so drop Content-Encoding or
        # the rebuilt response would try to decode it a second time.
        headers = [
            (key, value)
            for key, value in response.headers.multi_items()
            if key.lower() != "content-encoding"
        ]
        return httpx.Response(
            response.status_code,
            headers=headers,
            content=b"".join(chunks),
            request=response.request,
        )


def bounded_read(
    response: urllib.response.addinfourl, *, max_bytes: int = _MAX_BODY_BYTES
) -> bytes:
    """read() a urlopen remote response with a cap — the server can answer
    with an unbounded body, so read at most `max_bytes` + 1 and fail."""
    body = response.read(max_bytes + 1)
    if len(body) > max_bytes:
        raise ValueError("upstream response exceeds the fetch body limit")
    return body
