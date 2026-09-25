from __future__ import annotations

import argparse

import uvicorn

from . import create_app


def _port(value: str) -> int:
    parsed = int(value)
    if parsed < 1 or parsed > 65535:
        raise argparse.ArgumentTypeError("must be a port between 1 and 65535")
    return parsed


def serve(port: int = 18000) -> None:
    uvicorn.run(create_app(watchdog=True), host="127.0.0.1", port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=_port, default=18000)
    serve(parser.parse_args().port)
