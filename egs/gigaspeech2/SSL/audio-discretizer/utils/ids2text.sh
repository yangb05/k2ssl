#!/bin/bash

feats_scp="data/ml_superb_whisper_large_v3_kms2000/feats.scp"
output="$(dirname ${feats_scp})/output_quantized"

echo "feats_scp: ${feats_scp}"
echo "output: ${output}"

python utils/ids2text.py scp:${feats_scp} ${output}
