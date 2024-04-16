num=2000
feats_scp="data/ml_superb_whisper_large_v3/feats.scp"
output_dir="data/ml_superb_whisper_large_v3_kms2000"

. utils/parse_options.sh || exit 1

echo "num: ${num}"
echo "feats_scp: ${feats_scp}"
echo "output_dir: ${output_dir}"
mkdir -p ${output_dir}

model_path="${output_dir}/kmeans.bin"
log_path="${output_dir}/train_kms.log"
echo "model_path: ${model_path}"
echo "log_path: ${log_path}"


utils/run.pl ${log_path} \
    python kmeans/train_kmeans_partially.py \
        --feats-scp $feats_scp \
        --model-path $model_path \
        --n-clusters $num \
        --percent 1.0 \
        --batch-size 10000 \
        --init "k-means++"
