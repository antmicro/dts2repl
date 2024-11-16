#!/bin/bash

RED="\\033[1;31m"
GREEN="\\033[1;32m"
NC="\\033[0m"
COLOR=$NC
MSG=""
RESULT=0

echo -e "\n\n-----------------------------------------------------------"
echo "Status changes:"
echo "-----------------------------------------------------------"
if [[ ! -d ci-output/status || "$(find ci-output/status -name *.dts | wc -l)" == 0 ]] ; then
    echo "  NO STATUS CHANGES."
    echo -e "-----------------------------------------------------------\n\n"
    exit $RESULT
fi
cd ci-output/status
for dir in */ ; do
    dirname=${dir::-1}
    if [ "$dirname" == "loading-to-not_loading" ] ; then
        MSG="LOADING -> NOT LOADING"
        COLOR=$RED
        #RESULT=1
    elif [ "$dirname" == "not_existing-to-loading" ] ; then
        MSG="NOT EXISTING -> LOADING"
        COLOR=$GREEN
    elif [ "$dirname" == "not_existing-to-not_loading" ] ; then
        MSG="NOT EXISTING -> NOT LOADING"
        COLOR=$GREEN
    elif [ "$dirname" == "not_loading-to-loading" ] ; then
        MSG="NOT LOADING -> LOADING"
        COLOR=$GREEN
    elif [ "$dirname" == "not_loading-to-not_existing" ] ; then
        MSG="NOT LOADING -> NOT EXISTING"
        COLOR=$RED
    elif [ "$dirname" == "loading-to-not_existing" ] ; then
        MSG="LOADING -> NOT EXISTING"
        COLOR=$RED
    fi
    cd $dir
    for subdir in */ ; do
        subdirname=${subdir::-1}
        echo -e "  ${subdirname}: ${COLOR}${MSG}${NC}"
    done
    cd ..
done
echo -e "-----------------------------------------------------------\n\n"

cp -R ./passed-to-generated ../regressions 2> /dev/null || true
exit $RESULT
