#!/bin/bash
set -e

# config
feats_scp="data/ml_superb_whisper_large_v3/feats.scp"
output_dir="data/ml_superb_whisper_large_v3_kms2000"
km_model_path="data/ml_superb_whisper_large_v3_kms2000/kmeans.bin"

# parallel
nj=4

. utils/parse_options.sh
mkdir -p ${output_dir}
echo "feats_scp: ${feats_scp}"
echo "output_dir: ${output_dir}"
echo "km_model_path: ${km_model_path}"
echo "nj: ${nj}"

export PYTHONPATH=$PWD:$PYTHONPATH

for i in $(seq 1 ${nj}); do
    mkdir -p ${output_dir}/split.${i}
    utils/split_scp.pl -j ${nj} ${i} --one-based ${feats_scp} > ${output_dir}/split.${i}/feats.scp
done

utils/run.pl JOB=1:${nj} ${output_dir}/split.JOB/decode_kms.log \
    python kmeans/decode_kmeans.py \
        --feats-scp ${output_dir}/split.JOB/feats.scp \
        --output-dir ${output_dir}/split.JOB \
        --kmeans $km_model_path 

cat ${output_dir}/split.*/feats.scp | sort > ${output_dir}/feats.scp
