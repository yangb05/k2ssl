#!/bin/bash
set -e

utt2duration="/mnt/lustre/sjtu/home/fys18/dataset/Aishell/kaldi_style/train/utt2duration"
outdir="data/aishell1-100h"
wav_scp="data/aishell1/wav.scp"
min=0
max=20
target=360000   # 100*3600

cat ${utt2duration} | shuf | awk -v min=${min} -v max=${max} -v target=${target} 'BEGIN{S=0;}{
if($2>=min&&$2<=max&&S<target) {
    S+=$2;
    print $1;
}
}' > ${outdir}/uttids

cat ${utt2duration} | grep -Ff ${outdir}/uttids > ${outdir}/utt2duration

# summarize
echo "filtered $(cat ${outdir}/utt2duration | wc -l) lines."
filtered_duration=$(cat ${outdir}/utt2duration | awk '{S+=$2;}END{print S/3600}')
echo "filtered ${filtered_duration} hours"

# wav_scp="$(dirname ${utt2duration})/wav.scp"
if [ -f ${wav_scp} ];then
    echo "filtering wav.scp..."
    cat ${wav_scp} | grep -Ff ${outdir}/uttids > ${outdir}/wav.scp
fi

# feats_scp="$(dirname ${utt2duration})/feats.scp"
# if [ -f ${feats_scp} ];then
#     echo "filtering feats.scp..."
#     cat ${feats_scp} | grep -Ff ${outdir}/uttids > ${outdir}/feats.scp
# fi

echo "finished"
