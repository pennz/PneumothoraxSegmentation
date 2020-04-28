from subprocess import call


with open("runner.sh", "w") as f:
    f.write(
        """#!/bin/bash
USER=$1
shift
REPO=$1
shift
BRANCH=$1
shift
PHASE=$1
shift
PARAMS=$@

( test -d ${REPO} || git clone --depth=1 \
https://github.com/${USER}/${REPO}.git ) && cd ${REPO} && \
git checkout -b _${BRANCH} --track origin/${BRANCH} && \
( [[ x"$PHASE" == x"dev" ]]  && pytest ) && python main.py $PARAMS
"""
    )
call(
    [
        "bash",
        "-x",
        "runner.sh",
        "pennz",
        "PneumothoraxSegmentation",
        "master",
        "dev",
        "other",
        "paras",
    ]
)
