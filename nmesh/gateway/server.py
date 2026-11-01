from __future__ import annotations

import argparse

import uvicorn

from nmesh.net import port_arg

from . import create_app


def serve(port: int = 18000) -> None:
    uvicorn.run(create_app(watchdog=True), host="127.0.0.1", port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=port_arg, default=18000)
    serve(parser.parse_args().port)
