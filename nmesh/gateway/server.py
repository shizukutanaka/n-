from __future__ import annotations

import argparse

import uvicorn

from . import create_app


def _port(text: str) -> int:
    try:
        parsed = int(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if not 1 <= parsed <= 65535:
        raise argparse.ArgumentTypeError("must be between 1 and 65535")
    return parsed


def serve(port: int = 18000) -> None:
    uvicorn.run(create_app(watchdog=True), host="127.0.0.1", port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=_port, default=18000)
    serve(parser.parse_args().port)
