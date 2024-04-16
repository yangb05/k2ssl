#!/bin/bash
set -e
set -o pipefail


head -n 10 data/librilight-100h/output/whisper/text  |  \
while read -r line; do
    uttid=$(echo ${line} | cut -d " " -f 1)
    text=$(echo ${line} | cut -d " " -f 2-)
    echo "${text}" > ../../AudioSubword/MFA/data/test/${uttid}.txt
done
