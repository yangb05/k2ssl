num=2000
batch_size=100000
cut_file="/mgData2/yangb/icefall/egs/vietnamese/ASR/data/fbank/vietnamese_cuts_train_no_perturb.jsonl.gz"
output_dir="data/vietnamese_fbank_kms2000_bs_100000"

. /mgData2/yangb/icefall-ssl/egs/gigaspeech2/SSL/audio-discretizer/utils/parse_options.sh || exit 1

echo "num: ${num}"
echo "batch_size: ${batch_size}"
echo "cut_file: ${cut_file}"
echo "output_dir: ${output_dir}"
mkdir -p ${output_dir}

model_path="${output_dir}/kmeans.bin"
log_path="${output_dir}/train_kms.log"
echo "model_path: ${model_path}"
echo "log_path: ${log_path}"


/mgData2/yangb/icefall-ssl/egs/gigaspeech2/SSL/audio-discretizer/utils/run.pl ${log_path} \
    python /mgData2/yangb/icefall-ssl/egs/gigaspeech2/SSL/audio-discretizer/kmeans/train_kmeans_partially_fbank.py \
        --cut-file $cut_file \
        --model-path $model_path \
        --n-clusters $num \
        --percent 1.0 \
        --batch-size $batch_size \
        --init "k-means++" \
        --layer-norm
