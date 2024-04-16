#!/bin/bash
set -e

# data
wavscp="data/ml_superb.scp"
layer=21
speed_ratio=1
outdir="$PWD/data/ml_superb_wavlm_large_l${layer}_kms2000_sp${speed_ratio}_nopad"
# config
ckpt_path="download/WavLM-Large.pt"
km_model_path="download/wavlm_large_l21_kms2000.bin"
# parallel
nj=4

. utils/parse_options.sh
mkdir -p ${outdir}
echo "wavscp: ${wavscp}"
echo "layer: ${layer}"
echo "outdir: ${outdir}"
echo "speed_ratio: ${speed_ratio}"
# echo "layernorm: ${layernorm}"
echo "ckpt_path: ${ckpt_path}"
echo "km_model_path: ${km_model_path}"
echo "nj: ${nj}"

# python path
export PYTHONPATH=$PWD:$PYTHONPATH

for i in $(seq 1 ${nj}); do
    mkdir -p ${outdir}/split.${i}
    utils/split_scp.pl -j ${nj} ${i} --one-based ${wavscp} > ${outdir}/split.${i}/wav.scp
done


# we set gpu device with arg: rank-1
utils/run.pl JOB=1:${nj} ${outdir}/split.JOB/encode.log \
    python wavlm/extract_index.py \
        --wavscp ${outdir}/split.JOB/wav.scp \
        --layer ${layer} \
        --speed-ratio ${speed_ratio} \
        --outdir ${outdir}/split.JOB \
        --model $ckpt_path \
        --kmeans $km_model_path \
        --rank JOB # ${layer_norm} 

cat ${outdir}/split.*/feats.scp | sort > ${outdir}/feats.scp
