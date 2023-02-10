#!/bin/bash
set -euo pipefail

# Dashboard URL in argument overrides environment
DASHBOARD_URL="${1:-${DASHBOARD_URL:-}}"
DASHBOARD_URL="${DASHBOARD_URL:?Please provide a dashboard URL}"

curl -Lsf "$DASHBOARD_URL/replkit.tar.xz" -o replkit.tar.xz
mkdir replkit
tar xf replkit.tar.xz -C replkit >/dev/null
# Grab only the hello_world repls and dtses
# We assume they will be the same as for other samples
rename -E 's:replkit/:dts/:' -E 's:-hello_world.dts:.dts:' replkit/*-hello_world.dts
rename -E 's:replkit/:repls/dashboard/:' -E 's:-hello_world.repl:.repl:' replkit/*-hello_world.repl
# Remove the leftovers (dts/repl from other samples)
rm replkit/*.{dts,repl}
# Grab the versions
mv replkit/*.version .

rmdir replkit 2>/dev/null || {
    echo replkit had unexpected files:
    ls -l replkit
}
