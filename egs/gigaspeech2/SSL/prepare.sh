#!/usr/bin/env bash

# fix segmentation fault reported in https://github.com/k2-fsa/icefall/issues/674
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

set -eou pipefail

nj=16
# run step 1 to step 5 by default
stage=1
stop_stage=5

# We assume dl_dir (download dir) contains the following directories and files.
#
#  - $dl_dir/GigaSpeech2

dl_dir=$PWD/download
lang=Thai
num_per_split=20000

. shared/parse_options.sh || exit 1

# All files generated by this script are saved in "data".
# You can safely remove "data" and rerun this script to regenerate it.
mkdir -p data

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

log "Running prepare.sh"

log "dl_dir: $dl_dir"

subsets=""
for dir in ${dl_dir}/GigaSpeech2/* ; do
  subset=$(basename $dir)
  subsets="$subsets $subset"
done
log "Find subsets: $subsets"

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Prepare GigaSpeech2 manifest, language: $lang"
  # We assume that you have downloaded the GigaSpeech2 corpus
  # to $dl_dir/GigaSpeech2
  mkdir -p data/manifests
  if [ ! -e data/manifests/.gigaspeech2.done ]; then
    lhotse prepare gigaspeech2 --lang $lang -j $nj $dl_dir/GigaSpeech2 data/manifests
    touch data/manifests/.gigaspeech2.done
  fi
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "State 2: Preprocess GigaSpeech2 manifest"
  if [ ! -f data/fbank/.preprocess.done ]; then
   python3 ./local/preprocess_gigaspeech2.py --lang $lang --dataset "$subsets"
   touch data/fbank/.preprocess.done
  fi
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "Stage 3: Compute fbank for test set"
  mkdir -p data/fbank
  ./local/compute_fbank_gigaspeech2.py
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  log "Stage 4: Split train set into pieces"
  for subset in $subsets; do
    if [[ $subset != "test" ]]; then
      log "Split subset: $subset"
      split_dir=data/fbank/${subset}_split
      if [ ! -f $split_dir/.split.done ]; then
        lhotse split-lazy ./data/fbank/gigaspeech2_cuts_${subset}_raw.jsonl.gz $split_dir $num_per_split
        touch $split_dir/.split.done
      fi
    fi
  done
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
  log "Stage 5: Compute features for train set"
  for subset in $subsets; do
    if [[ $subset != "test" ]]; then
      log "Compute features for subset: $subset"
      split_dir=data/fbank/${subset}_split
      num_splits=$(find $split_dir -name "gigaspeech2_cuts_${subset}_raw.*.jsonl.gz" | wc -l)
      python3 ./local/compute_fbank_gigaspeech2_splits.py \
	--dataset $subset \
        --num-workers 20 \
        --batch-duration 1000 \
        --num-splits $num_splits
    fi
  done
fi