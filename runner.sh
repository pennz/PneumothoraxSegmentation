#!/bin/bash
USER=$1
shift
REPO=$1
shift
PHASE=$1
shift
PARAMS=$@

( test -d ${REPO} || git clone --depth=1 https://github.com/${USER}/${REPO}.git ) && cd ${REPO} && ( [[ x"$PHASE" == x"dev" ]]  && pytest ) && python main.py $PARAMS
