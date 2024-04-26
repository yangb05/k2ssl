#!/bin/bash
set -e

# config
cut_dir="/data_a100/userhome/yangb/data/fbank"
declare -a cut_files=("librispeech_cuts_dev-clean.jsonl.gz" "librispeech_cuts_dev-other.jsonl.gz" "librispeech_cuts_test-clean.jsonl.gz" "librispeech_cuts_test-other.jsonl.gz" "librispeech_cuts_train-clean-100.jsonl.gz" "librispeech_cuts_train-clean-360.jsonl.gz" "librispeech_cuts_train-other-500.jsonl.gz")
kmeans="/mgData2/yangb/icefall-ssl/egs/gigaspeech2/SSL/data/librispeech_embed_kms500/kmeans.bin"
output_dir="/data_a100/userhome/yangb/data/librispeech_embed_kms500"
mkdir -p $output_dir
num_workers=20

. /mgData2/yangb/icefall-ssl/egs/gigaspeech2/SSL/audio-discretizer/utils/parse_options.sh
echo "cut_dir: ${cut_dir}"
echo "cut_files: ${cut_files}"
echo "kmeans: ${kmeans}"
echo "output_dir: ${output_dir}"
echo "num_workers: ${num_workers}"

for cut_file in ${cut_files[@]}
do
    embed_file=${cut_file/cuts/embed}
    embed_file=${embed_file/jsonl.gz/npy}
    python /mgData2/yangb/icefall-ssl/egs/gigaspeech2/SSL/audio-discretizer/kmeans/decode_kmeans_embed.py \
        --cut-file ${cut_dir}/${cut_file} \
        --embed-file ${cut_dir}/${embed_file} \
        --kmeans $kmeans \
        --output-file ${output_dir}/${cut_file} \
        --num-workers $num_workers \
        --layer-norm
done
