export CUDA_VISIBLE_DEVICES="2,3,4,5,6,7"

./zipformer/finetune.py \
  --world-size 6 \
  --num-epochs 50 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir zipformer/exp_finetune_vietnamese \
  --bpe-model data/lang_bpe_10000/bpe.model \
  --pretrained-dir zipformer/exp_pretrain_vietnamese/epoch-100.pt \
  --max-duration 600 \
  --accum-grad 1 \
  --mask-prob 0.65 \
  --mask-channel-prob 0.5 \
  --mask-channel-length 64 \
  --feature-grad-mult 0.0 \
  --num-encoder-layers 2,2,3,4,3,2 \
  --feedforward-dim 512,768,1024,1536,1024,768 \
  --encoder-dim 192,256,448,768,448,192 \
  --encoder-unmasked-dim 192,192,256,256,256,192 \
  --base-lr 0.002