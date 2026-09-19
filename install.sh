#!/usr/bin/env bash
# nmesh installer — creates a dedicated venv and installs nmesh with the
# gateway + download extras.
set -euo pipefail

NMESH_VENV="${NMESH_VENV:-$HOME/.nmesh/venv}"
REPO="${NMESH_REPO:-https://github.com/shizukutanaka/n-.git}"

# Xcode's python3 (3.9 on macOS) can shadow a newer interpreter such as
# Homebrew's python3.12, so probe versioned names too.
python_bin=""
for candidate in python3.13 python3.12 python3.11 python3.10 python3 python; do
    if bin="$(command -v "$candidate" 2>/dev/null)" \
        && "$bin" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
        python_bin="$bin"
        break
    fi
done
if [ -z "$python_bin" ]; then
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
