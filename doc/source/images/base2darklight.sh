#!/bin/bash

DARK_FILE="${1/base/dark}"
LIGHT_FILE="${1/base/light}"

sed -e 's/000000/333333/g' -e 's/ff0000/00a99d/g' ${1} > ${LIGHT_FILE}
sed -e 's/000000/cccccc/g' -e 's/ff0000/80d4ce/g' ${1} > ${DARK_FILE}

