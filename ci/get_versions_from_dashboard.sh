#!/bin/bash

if [[ "$DASHBOARD_VARIANT" == "zephyr" ]]; then
    export GCP_SIM_BASE="https://zephyr-dashboard.renode.io/zephyr_sim"
    export DEMO_NAME=hello_world
elif [[ "$DASHBOARD_VARIANT" == "uboot" ]]; then
    export GCP_SIM_BASE="https://u-boot-dashboard.renode.io/uboot_sim"
    export DEMO_NAME=uboot
else
    exit 1
fi

export LATEST_SIM=$(curl --fail -sS -G "$GCP_SIM_BASE"/latest)
export LATEST_SIM_RENODE=$(curl --fail -sS -G "$GCP_SIM_BASE"/"$LATEST_SIM"/latest)

if [ -z "$RENODE_VERSION" ]; then
    export RENODE_VERSION="$LATEST_SIM_RENODE"
fi
