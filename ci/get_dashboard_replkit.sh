#!/bin/bash
set -euo pipefail

if [[ "$DASHBOARD_VARIANT" == "zephyr" ]]; then
    GCP_SIM_BASE="https://zephyr-dashboard.renode.io/zephyr_sim"
    export DEMO_NAME=hello_world
elif [[ "$DASHBOARD_VARIANT" == "uboot" ]]; then
    GCP_SIM_BASE="https://u-boot-dashboard.renode.io/uboot_sim"
    export DEMO_NAME=uboot
else
    exit 1
fi

LATEST_SIM=$(curl --fail -sS -G "$GCP_SIM_BASE"/latest)
LATEST_RENODE=$(curl --fail -sS -G "$GCP_SIM_BASE"/"$LATEST_SIM"/latest)
wget "$GCP_SIM_BASE"/"$LATEST_SIM"/"$LATEST_RENODE"/replkit.tar.xz

mkdir replkit
tar xf replkit.tar.xz -C replkit >/dev/null

# Grab only the DEMO_NAME repls, configs and dtses
# We assume they will be the same as for other samples
rename -E 's:replkit/:dts/:' -E "s:-${DEMO_NAME}.dts:.dts:" replkit/*-${DEMO_NAME}.dts
rename -E 's:replkit/:repls/dashboard/:' -E "s:-${DEMO_NAME}.repl:.repl:" replkit/*-${DEMO_NAME}.repl

if ls replkit/*-"$DEMO_NAME"-config.json 1> /dev/null 2>&1; then
    rename -E 's:replkit/:repls/dashboard/:' -E "s:-${DEMO_NAME}-config.json:-config.json:" replkit/*-${DEMO_NAME}-config.json
    rm replkit/*.json 
fi

# Grab the versions
mv replkit/*.version .

# Remove the leftovers
rm replkit/*.repl
rm replkit/*.dts

rmdir replkit 2>/dev/null || {
    echo replkit had unexpected files:
    ls -l replkit
}
