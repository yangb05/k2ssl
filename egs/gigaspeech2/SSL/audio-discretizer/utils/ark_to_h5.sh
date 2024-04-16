#!/bin/bash

python utils/ark_to_h5.py \
    scp:data/aishell1/output/wavlm_large_l24_kms2000/feats.scp\
    data/aishell1/output/wavlm_large_l24_kms2000/feats.h5
