#!/bin/bash
set -e

# data
wavscp="data/ml_superb.scp"
layer=21
outdir="$PWD/data/ml_superb_wavlm_large_l${layer}"
# config
model="download/WavLM-Large.pt"

. utils/parse_options.sh
mkdir -p ${outdir}
echo "wavscp: ${wavscp}"
echo "layer: ${layer}"
echo "outdir: ${outdir}"
echo "model: ${model}"

# parallel
nj=4

for i in $(seq 1 ${nj}); do
    mkdir -p ${outdir}/split.${i}
    utils/split_scp.pl -j ${nj} ${i} --one-based ${wavscp} > ${outdir}/split.${i}/wav.scp
done

# we set gpu device with arg: rank-1
utils/run.pl JOB=1:${nj} ${outdir}/split.JOB/extract.log \
    python wavlm/extract_feature.py \
        --wavscp ${outdir}/split.JOB/wav.scp \
        --layer ${layer} \
        --outdir ${outdir}/split.JOB \
        --model ${model} \
        --rank JOB

cat ${outdir}/split.*/feats.scp | sort > ${outdir}/feats.scp
