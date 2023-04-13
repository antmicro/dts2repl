#!/bin/bash
set -euo pipefail

apt-get -qqy update
apt-get -qqy install git curl python3 python3-pip parallel xz-utils rename

# Install dts2repl
pip3 install .

# Install west
pip3 install west=="$WEST_VERSION"

# Prepare output directories
mkdir -p ci-output/{repls/{dashboard,generated,diffs},dts}
