export CUDA_VISIBLE_DEVICES="6"

./zipformer/finetune.py \
  --world-size 1 \
  --num-epochs 50 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir zipformer/exp_finetune_vietnamese_directly \
  --bpe-model data/lang_bpe_2000/bpe.model \
  --max-duration 2000 \
  --accum-grad 1 \
  --mask-prob 0.65 \
  --mask-channel-prob 0.5 \
  --mask-channel-length 64 \
  --feature-grad-mult 0.0 \
  --num-encoder-layers 2,2,3,4,3,2 \
  --feedforward-dim 512,768,1024,1536,1024,768 \
  --encoder-dim 192,256,448,768,448,192 \
  --encoder-unmasked-dim 192,192,256,256,256,192 \
  --base-lr 0.045 \
  --master-port 12356
