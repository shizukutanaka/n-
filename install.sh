#!/usr/bin/env bash
# nmesh installer — creates a dedicated venv and installs nmesh with the
# gateway + download extras.
set -euo pipefail

NMESH_VENV="${NMESH_VENV:-$HOME/.nmesh/venv}"
REPO="${NMESH_REPO:-https://github.com/shizukutanaka/n-.git}"

python_bin="$(command -v python3 || command -v python || true)"
if [ -z "$python_bin" ]; then
    echo "python3 is required (>= 3.10)" >&2
    exit 1
fi

if ! "$python_bin" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
    echo "python >= 3.10 is required" >&2
    exit 1
fi

"$python_bin" -m venv "$NMESH_VENV"
"$NMESH_VENV/bin/pip" install --upgrade pip
"$NMESH_VENV/bin/pip" install "nmesh[gateway,download] @ git+$REPO"

cat <<EOF

nmesh installed into $NMESH_VENV

Add it to your PATH:
  export PATH="$NMESH_VENV/bin:\$PATH"

Then:
  nmesh doctor      # check detected hardware and backends
  nmesh up --detach # pick models for this machine and start the gateway
EOF
