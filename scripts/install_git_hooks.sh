#!/usr/bin/env sh
set -eu

ROOT_DIR="$(git rev-parse --show-toplevel)"
cd "${ROOT_DIR}"

chmod +x .githooks/pre-push
git config core.hooksPath .githooks

echo "Installed repository hooks from .githooks/"
echo "Configured core.hooksPath=.githooks"
