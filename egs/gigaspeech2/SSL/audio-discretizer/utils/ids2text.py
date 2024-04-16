import sys

import numpy as np
from kaldiio import ReadHelper

rspec = sys.argv[1]
out = sys.argv[2]


with ReadHelper(rspec) as reader, open(out, "w") as writer:
    for uttid, feat in reader:
        # feat: (l, 1/2)
        feat = feat.reshape(-1).astype(np.int32)
        feat_str = " ".join(map(str, feat))

        writer.write("{} {}\n".format(uttid, feat_str))
