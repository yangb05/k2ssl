cut_file="librispeech_cuts_dev-clean.jsonl.gz"
embed_file=${cut_file/cuts/embed}
embed_file=${embed_file/jsonl.gz/npy}
echo $embed_file