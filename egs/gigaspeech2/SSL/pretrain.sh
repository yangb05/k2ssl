export CUDA_VISIBLE_DEVICES="2,3,4,5,6,7"

./zipformer/pretrain.py \
  --world-size 6 \
  --num-epochs 150 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir zipformer/exp_pretrain_vietnamese_mask_0.8_label_fixed_no_layernorm \
  --max-duration 1500 \
  --accum-grad 1 \
  --mask-prob 0.8 \
  --dropout-input 0.1 \
  --dropout-features 0.1 \
  --untie-final-proj 1 \
  --num-encoder-layers 2,2,3,4,3,2 \
  --feedforward-dim 512,768,1024,1536,1024,768 \
  --encoder-dim 192,256,448,768,448,192 \
  --encoder-unmasked-dim 192,192,256,256,256,192 \
  --base-lr 0.045