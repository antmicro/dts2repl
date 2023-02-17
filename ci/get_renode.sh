#!/bin/bash
set -euo pipefail

if [ -d renode-portable ]; then
    echo Renode already downloaded >&2
    exit
fi

# Renode version in argument overrides environment
RENODE_VERSION="${1:-${RENODE_VERSION:-}}"
RENODE_VERSION="${RENODE_VERSION:?Please provide a Renode version}"

mkdir -p renode-portable
curl -Lsf "https://builds.renode.io/renode-${RENODE_VERSION}.linux-portable.tar.gz" | tar xz -C renode-portable --strip-components=1
pip3 install -r renode-portable/tests/requirements.txt
