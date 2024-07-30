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
    if [ "$dirname" == "passed-to-generated" ] ; then
        MSG="PASSED -> GENERATED"
        COLOR=$RED
        RESULT=1
    elif [ "$dirname" == "not_generated-to-passed" ] ; then
        MSG="NOT GENERATED -> PASSED"
        COLOR=$GREEN
    elif [ "$dirname" == "not_generated-to-generated" ] ; then
        MSG="NOT GENERATED -> GENERATED"
        COLOR=$GREEN
    elif [ "$dirname" == "generated-to-passed" ] ; then
        MSG="GENERATED -> PASSED"
        COLOR=$GREEN
    fi
    cd $dir
    for subdir in */ ; do
        subdirname=${subdir::-1}
        echo -e "  ${subdirname}: ${COLOR}${MSG}${NC}"
    done
    cd ..
done
echo -e "-----------------------------------------------------------\n\n"

cp -R ./passed-to-generated ../regressions
exit $RESULT
