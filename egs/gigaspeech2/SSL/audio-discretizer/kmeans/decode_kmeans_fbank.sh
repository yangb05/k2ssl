#!/bin/bash
set -e

# config
cut_dir="/mgData2/yangb/icefall-ssl/egs/gigaspeech2/SSL/data/fbank"
declare -a cut_files=("vietnamese_cuts_train_no_perturb.jsonl.gz" "vietnamese_cuts_dev.jsonl.gz" "vietnamese_cuts_test.jsonl.gz")
kmeans="/mgData2/yangb/icefall-ssl/egs/gigaspeech2/SSL/data/vietnamese_fbank_kms2000_bs_100000/kmeans.bin"
output_dir="/data_a100/userhome/yangb/data/fbank/viet_fbank_kms2000_100000"
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
    python /mgData2/yangb/icefall-ssl/egs/gigaspeech2/SSL/audio-discretizer/kmeans/decode_kmeans_fbank.py \
        --cut-file ${cut_dir}/${cut_file} \
        --kmeans $kmeans \
        --output-file ${output_dir}/${cut_file} \
        --num-workers $num_workers \
        --layer-norm
done
