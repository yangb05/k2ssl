export CUDA_VISIBLE_DEVICES=1
for ((epoch=20; epoch<=50; epoch+=3)); do
  for ((avg=10; avg<=$epoch; avg+=3)); do
    ./zipformer/decode.py \
      --exp-dir zipformer/exp_finetune_vietnamese_284h_label_fixed_pretrain_best-valid_mask_0.8_bs_2000_lr_0.01 \
      --epoch $epoch \
      --avg $avg \
      --num-encoder-layers 2,2,3,4,3,2 \
      --feedforward-dim 512,768,1024,1536,1024,768 \
      --encoder-dim 192,256,448,768,448,192 \
      --encoder-unmasked-dim 192,192,256,256,256,192 \
      --bpe-model data/lang_bpe_2000/bpe.model \
      --max-duration 2000 \
      --decoding-method greedy_search
  done
done