export CUDA_VISIBLE_DEVICES="1"

./zipformer/finetune.py \
  --world-size 1 \
  --num-epochs 100 \
  --lr-epochs 35 \
  --lr-batches 5000 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir zipformer/exp_finetune_vietnamese_284h_label_fixed_pretrain_best-valid_mask_0.8_lr_0.045 \
  --bpe-model data/lang_bpe_2000/bpe.model \
  --pretrained-dir zipformer/exp_pretrain_vietnamese_mask_0.8_label_fixed/best-valid-loss.pt \
  --max-duration 2000 \
  --accum-grad 1 \
  --feature-grad-mult 0.0 \
  --num-encoder-layers 2,2,3,4,3,2 \
  --feedforward-dim 512,768,1024,1536,1024,768 \
  --encoder-dim 192,256,448,768,448,192 \
  --encoder-unmasked-dim 192,192,256,256,256,192 \
  --base-lr 0.045 \
  --master-port 12358