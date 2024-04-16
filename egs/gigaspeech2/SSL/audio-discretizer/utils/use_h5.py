import h5py
import numpy as np

with h5py.File("data/gigaspeech-1000h/output/encodec/feats.h5", "r") as reader:
    uttids = sorted(reader.keys())
    for uttid in uttids:
        ids = np.array(reader[uttid])
        print(ids.shape)
        print(ids.dtype)
        break
