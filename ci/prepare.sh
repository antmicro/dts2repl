#!/bin/bash

# checkout on the specified version if needed
if [ -n "$DTS2REPL_VERSION" ]; then
	echo -e "\n========== DOING CHECKOUT ON VERSION ============"
	echo $DTS2REPL_VERSION
	git checkout $DTS2REPL_VERSION
fi
echo -e "\n========== INSTALLING DTS2REPL VERSION =========="
git branch -v
echo -e "=================================================\n"

# install dts2repl as module
pip3 install .

# download replkit and prepare dts/repl directory structure
mkdir -p ci-output/{repls/{dashboard,generated,diffs},dts}
(cd ci-output && ../ci/get_dashboard_replkit.sh)

# download or build Renode
./ci/get_renode.sh
