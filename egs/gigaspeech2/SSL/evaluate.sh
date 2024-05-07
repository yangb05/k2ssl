export CUDA_VISIBLE_DEVICES=1
for epoch in 30 35 40 45 50; do
  for avg in 5 10 15 20; do
    ./zipformer/decode.py \
      --epoch $epoch \
      --avg $avg \
      --num-encoder-layers 2,2,3,4,3,2 \
      --feedforward-dim 512,768,1024,1536,1024,768 \
      --encoder-dim 192,256,448,768,448,192 \
      --encoder-unmasked-dim 192,192,256,256,256,192 \
      --exp-dir ./zipformer/exp_finetune_vietnamese \
      --bpe-model data/lang_bpe_10000/bpe.model \
      --max-duration 1500 \
      --decoding-method greedy_search
  done
done