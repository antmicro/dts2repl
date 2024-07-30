#!/bin/bash

# install dts2repl as module
pip3 install .

# download replkit and prepare dts/repl directory structure
mkdir -p ci-output/{repls/{dashboard,generated,diffs},dts}
(cd ci-output && ../ci/get_dashboard_replkit.sh)

# download or build Renode
./ci/get_renode.sh
