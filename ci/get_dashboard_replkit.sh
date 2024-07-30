#!/bin/bash
set -euo pipefail

GCP_SIM_BASE="https://zephyr-dashboard.renode.io/zephyr_sim"
LATEST_ZEPHYR=$(curl --fail -sS -G "$GCP_SIM_BASE"/latest)
LATEST_RENODE=$(curl --fail -sS -G "$GCP_SIM_BASE"/"$LATEST_ZEPHYR"/latest)

wget "$GCP_SIM_BASE"/"$LATEST_ZEPHYR"/"$LATEST_RENODE"/replkit.tar.xz
mkdir replkit
tar xf replkit.tar.xz -C replkit >/dev/null
# Grab only the hello_world repls, configs and dtses
# We assume they will be the same as for other samples
rename -E 's:replkit/:dts/:' -E 's:-hello_world.dts:.dts:' replkit/*-hello_world.dts
rename -E 's:replkit/:repls/dashboard/:' -E 's:-hello_world.repl:.repl:' replkit/*-hello_world.repl
rename -E 's:replkit/:repls/dashboard/:' -E 's:-hello_world-config.json:-config.json:' replkit/*-hello_world-config.json
# Remove the leftovers (dts/repl/json from other samples)
rm replkit/*.repl
rm replkit/*.json
rm replkit/*.dts
# Grab the versions
mv replkit/*.version .

rmdir replkit 2>/dev/null || {
    echo replkit had unexpected files:
    ls -l replkit
}
