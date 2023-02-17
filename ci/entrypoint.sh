#!/bin/bash
set -euxo pipefail

cd ci-output

ZEPHYR_VERSION="$(<zephyr.version)"
echo "Zephyr version is $ZEPHYR_VERSION"

# Download Zephyr if not already present
../ci/get_zephyr.sh "$ZEPHYR_VERSION"

# Save board list to use in later steps
# Assumption: board names and architectures do not contain spaces
(cd zephyrproject/zephyr; west boards --format='{arch} {name} {dir}') > boards

# Remove boards that don't have dashboard dtses from the list
set +x
while read -r arch board rest; do
    if [ -f "dts/${board}.dts" ]; then
        echo "$arch $board $rest"
    fi
done < boards > boards_filtered
set -x
mv boards_filtered boards

# Generate repls for the boards that have downloaded dtses for comparison
# Also fill in any missing dashboard repls with empty files
(
    cd repls/generated
    parallel --colsep ' ' '../../../ci/zephyrboard2repl.py --zephyr-path=../../zephyrproject/zephyr --flat-dts=../../dts/{2}.dts {} > {2}.repl; touch ../dashboard/{2}.repl' '::::' ../../boards
)

# Save diffs for repls that are different
# diff returns 0 on no difference which we use to delete diff files if the
# generated repl is identical (+/- blank lines) to the dashboard one
cd repls/diffs
parallel --colsep ' ' 'diff -u --ignore-blank-lines "../dashboard/{2}.repl" "../generated/{2}.repl" > "{2}.repl" && rm "{2}.repl"' '::::' ../../boards || true
# Print diffs to job output
# tail is used to prefix each file with its name, -n+3 will display the diff
# skipping the ---/+++ header
echo Listing repls with differences
tail -n+3 -v * || {
    echo No differences, not running Renode
    exit
}
cd -

# Download the Renode portable package and install its Python dependencies
../ci/get_renode.sh

# Try to load the repls that had differences in Renode
./renode-portable/renode-test --results-dir robot-results "$PWD/../ci/load_repls_with_diffs.robot"
