#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
export LD_LIBRARY_PATH="$PWD/vendor/system-libs/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec /home/a/download/lerobot/.venv/bin/python app.py "$@"
