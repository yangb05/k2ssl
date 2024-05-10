num=500
batch_size=100000
embed_files="/data_a100/userhome/yangb/data/fbank/vietnamese_embed_train_no_perturb.npy"
output_dir="/mgData2/yangb/icefall-ssl/egs/gigaspeech2/SSL/data/vietnamese_embed_kms500_bs_100000"

. /mgData2/yangb/icefall-ssl/egs/gigaspeech2/SSL/audio-discretizer/utils/parse_options.sh || exit 1

echo "num: ${num}"
echo "batch_size: ${batch_size}"
echo "embed_files: ${embed_files}"
echo "output_dir: ${output_dir}"
mkdir -p ${output_dir}

model_path="${output_dir}/kmeans.bin"
log_path="${output_dir}/train_kms.log"
echo "model_path: ${model_path}"
echo "log_path: ${log_path}"


/mgData2/yangb/icefall-ssl/egs/gigaspeech2/SSL/audio-discretizer/utils/run.pl ${log_path} \
    python /mgData2/yangb/icefall-ssl/egs/gigaspeech2/SSL/audio-discretizer/kmeans/train_kmeans_partially_embed.py \
        --embed-files $embed_files \
        --model-path $model_path \
        --n-clusters $num \
        --percent 1.0 \
        --batch-size $batch_size \
        --init "k-means++"
