import sys

import h5py
import numpy as np
from kaldiio import ReadHelper

if __name__ == "__main__":
    rspec = sys.argv[1]
    out = sys.argv[2]
    with ReadHelper(rspec) as reader, h5py.File(out, "w") as writer:
        for uttid, ids in reader:
            ids = ids.astype(np.int64)
            writer.create_dataset(uttid, data=ids)
