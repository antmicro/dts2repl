#!/usr/bin/env bash

try_download() {
    if [ $RENODE_VERSION = "latest" ]; then
        echo "===[Trying to obtain the latest Renode version from the Github repository]==="
        RENODE_VERSION=$(git ls-remote https://github.com/renode/renode refs/heads/master | cut -f1)
        if [ -z "$RENODE_VERSION" ]; then
            echo "Failed to obtain the latest Renode version from the Github repository!"
            return 1
        fi
        echo "Latest Renode version is: ${RENODE_VERSION}"
    fi

    echo "===[Trying to obtain precompiled package for renode commit ${RENODE_VERSION}]==="
    VERSION=$(curl -kLsf https://raw.githubusercontent.com/renode/renode/${RENODE_VERSION}/tools/version)
    if [ $? -ne 0 ]; then
        echo "No package with version ${RENODE_VERSION} found! (failed to resolve Renode version)"
        return 1
    fi

    curl -kLsf -o 'renode-portable.tar.gz' https://dl.antmicro.com/projects/renode/builds/sha/renode-"$VERSION"+git"${RENODE_VERSION:0:9}".linux-portable.tar.gz

    if [ $? -eq 0 ]; then
        echo "Precompiled package for renode commit ${RENODE_VERSION} downloaded."
        tar xf renode*portable.tar.gz -C ..
        cd ..
        mv renode_*_portable renode-portable
        RENODE_LOCATION=$(realpath renode-portable)
        echo "Renode portable is now in ${RENODE_LOCATION}"
        echo ${RENODE_LOCATION} > ${CI_PROJECT_DIR}/renode-location
        pip install -r renode-portable/tests/requirements.txt
        return 0
    else
        echo "No package with version ${RENODE_VERSION} found! (failed to download package)"
        return 1
    fi
}

try_build() {
    echo "===[Trying to build Renode from sources]==="
    if [ -n "$GET_CUSTOM_RENODE_REVISION" ]; then
        eval $GET_CUSTOM_RENODE_REVISION
    else
        git clone https://github.com/renode/renode.git
    fi

    cd renode
    RENODE_LOCATION=$(realpath .)
    echo "Renode sources are now in ${RENODE_LOCATION}"
    echo ${RENODE_LOCATION} > ${CI_PROJECT_DIR}/renode-location
    git checkout "${RENODE_VERSION}" || exit 1
    git submodule update --init --recursive || exit 1

    git fetch --all 1>/dev/null 2>/dev/null
    RENODE_HASH=$(git rev-parse HEAD)

    git branch -a --contains "$RENODE_VERSION" | grep 'renode_github'
    if [ $? -ne 0 ]; then
        printf "\033[0;31mRENODE COMMIT %s IS *NOT* PRESENT ON THE PUBLIC GITHUB\033[0m\n" "$RENODE_HASH"
    fi

    git branch -r --contains "$RENODE_HASH" | grep 'renode_github/master'
    if [ $? -eq 0 ]; then
        PUBLIC_MASTER=true
        printf "\033[0;32mRENODE COMMIT %s IS PRESENT ON THE PUBLIC GITHUB MASTER BRANCH\033[0m\n" "$RENODE_HASH"
    fi

    ./build.sh 1>../renode-build.log 2>&1 || return $?
    pip install -r tests/requirements.txt

    cd ..
    return 0
}

if try_download; then
    echo "==[Downloaded Renode from packages]==="
    exit 0
fi

if try_build; then
    echo "===[Built Renode from source]==="
    touch "renode_from_source"
    if [ "$PUBLIC_MASTER" = true ]; then touch "renode_packageable_revision"; fi
    exit 0
fi

echo "Unable to find or build a package for Renode version ${RENODE_VERSION}"
exit 1
