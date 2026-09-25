"""HTTP helpers for endpoints that are always local (127.0.0.1).

All nmesh services, the gateway, and the Ollama daemon bind to loopback.
Environment proxy variables (HTTP_PROXY/HTTPS_PROXY/ALL_PROXY) can never
serve a loopback address — a proxy lives on another host — yet both urllib
and httpx honor them by default, so a machine configured for a corporate
proxy silently routes every health probe and upstream call through a dead
end. The helpers below bypass env proxies for those known-local targets;
remote endpoints (model downloads, watch sources, delegate workers) keep
honoring the user's proxy settings.
"""

from __future__ import annotations

import urllib.request
import urllib.response
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import httpx

_LOCAL_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def local_urlopen(
    request: urllib.request.Request | str, *, timeout: float
) -> urllib.response.addinfourl:
    """urlopen that ignores env proxy settings; for loopback endpoints."""
    return _LOCAL_OPENER.open(request, timeout=timeout)


def local_client(
    timeout: httpx.Timeout | float | None = None,
    *,
    base_url: str = "",
    follow_redirects: bool = False,
) -> httpx.Client:
    """httpx.Client that ignores env proxy settings (httpx is optional)."""
    import httpx

    return httpx.Client(
        timeout=timeout, base_url=base_url, trust_env=False,
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
        timeout=timeout, base_url=base_url, trust_env=False,
        follow_redirects=follow_redirects,
    )
