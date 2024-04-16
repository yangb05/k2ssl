#!/bin/bash
set -e

# config
wav_scp="data/ml_superb.scp"
output_dir="$PWD/data/ml_superb_whisper_large_v3"

# parallel
nj=4

. utils/parse_options.sh
mkdir -p ${output_dir}
echo "wav_scp: ${wav_scp}"
echo "output_dir: ${output_dir}"
echo "nj: ${nj}"

for i in $(seq 1 ${nj}); do
    mkdir -p ${output_dir}/split.${i}
    utils/split_scp.pl -j ${nj} ${i} --one-based ${wav_scp} > ${output_dir}/split.${i}/wav.scp
done

utils/run.pl JOB=1:${nj} ${output_dir}/split.JOB/extract.log \
    python whisper/extract_feature.py \
        --wav-scp ${output_dir}/split.JOB/wav.scp \
        --output-dir ${output_dir}/split.JOB \
        --rank JOB

cat ${output_dir}/split.*/feats.scp | sort > ${output_dir}/feats.scp
