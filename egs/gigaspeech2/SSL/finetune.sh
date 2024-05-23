export CUDA_VISIBLE_DEVICES="7"

./zipformer/finetune.py \
  --world-size 1 \
  --num-epochs 60 \
  --lr-epochs 35 \
  --lr-batches 5000 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir zipformer/exp_finetune_vietnamese_284h_pretrain_epoch_90_mask_0.65_lr_0.017 \
  --bpe-model data/lang_bpe_2000/bpe.model \
  --pretrained-dir /data_a100/userhome/yangb/data/checkpoints/vietnamese_SSL/exp_pretrain_vietnamese_mask_0.65/epoch-90.pt \
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
  --base-lr 0.017 \
  --master-port 12358