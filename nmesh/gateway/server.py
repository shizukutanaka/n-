from __future__ import annotations

import uvicorn

from . import create_app


def serve() -> None:
    uvicorn.run(create_app(), host="127.0.0.1", port=18000)
