#!/bin/bash
set -e

# data
wavdir=""
wavscp="data/librilight-100h/wav.scp"
out="data/librilight-100h/output/whisper/raw.out"

. utils/parse_options.sh

tmpdir=$(mktemp -d tmp.XXX)

# get wav.scp from wavdir
if [ -z ${wavscp} ]; then
    wavscp="$(dirname ${out})/wav.scp"
    echo "generate wav.scp(${wavscp}) from ${wavdir}"
    ls ${wavdir} | awk -v dir=${wavdir} -F '.' '{printf "%s %s/%s\n", $1, dir, $0}' > ${wavscp}
fi


# parallel
cuda_devices=$(echo $CUDA_VISIBLE_DEVICES)
num_cuda_devices=$(echo ${cuda_devices} | awk -F "," "{print NF}")
echo "num_cuda_devices: ${num_cuda_devices}"
NJ=${num_cuda_devices}


for i in $(seq 1 ${NJ}); do
    mkdir -p ${tmpdir}/split.${i}
    utils/split_scp.pl -j ${NJ} ${i} --one-based ${wavscp} > ${tmpdir}/split.${i}/wav.scp
done


# we set gpu device with arg: rank-1
utils/run.pl JOB=1:${NJ} ${tmpdir}/split.JOB/whisper.log \
    python utils/whisper_asr.py \
        --wavscp ${tmpdir}/split.JOB/wav.scp \
        --rank JOB \> ${tmpdir}/split.JOB/raw.out

cat ${tmpdir}/split.*/raw.out | sort > ${out}

# rm -r ${tmpdir}
