#!/bin/bash
set -euxo pipefail

cd ci-output

# Generate repls for the boards that have downloaded dtses for comparison
(
    cd repls/generated
    ls -1 ../../dts | cut -d. -f1 | parallel 'dts2repl ../../dts/{}.dts --output {}.repl'
)

# Save diffs for repls that are different
# diff returns 0 on no difference which we use to delete diff files if the
# generated repl is identical (+/- blank lines) to the dashboard one
cd repls/diffs
ls -1 ../generated | parallel 'diff -u --new-file --ignore-blank-lines "../dashboard/{}" "../generated/{}" > "{}.diff" && rm "{}.diff" || true'
# Print diffs to job output
# tail is used to prefix each file with its name, -n+3 will display the diff
# skipping the ---/+++ header
echo Listing repls with differences
tail -n+3 -v * && DIFF=1
cd ../../

if [ "${DIFF}" -eq 1 ] ; then
    echo Found differences, running Renode
    ../renode-portable/renode-test --results-dir robot-results "$PWD/../ci/load_repls_with_diffs.robot"
fi
