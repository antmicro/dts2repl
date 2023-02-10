#!/bin/bash
set -euo pipefail

if [ -d zephyrproject/zephyr ]; then
    echo Zephyr already downloaded >&2
    exit
fi

# Zephyr version in argument overrides environment
ZEPHYR_VERSION="${1:-${ZEPHYR_VERSION:-}}"
ZEPHYR_VERSION="${ZEPHYR_VERSION:?Please provide a Zephyr branch/tag/commit (for example main)}"

mkdir -p zephyrproject/zephyr
cd zephyrproject/zephyr || exit 1

git init &>/dev/null
git remote add origin https://github.com/zephyrproject-rtos/zephyr
git pull --depth 1 origin "$ZEPHYR_VERSION"
git reset --hard

west init -l .
