#!/usr/bin/env bash
# Install the locked environment, but with the CPU build of the locked torch version instead of PyPI's CUDA build.
set -euo pipefail
torch=$(sed -n '/^name = "torch"$/{n;s/^version = "\(.*\)"$/\1/p;}' uv.lock)
uv sync --locked --all-extras --no-install-package torch
uv pip install "torch==$torch" --index-url https://download.pytorch.org/whl/cpu
