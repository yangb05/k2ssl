import argparse
import logging
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from collections import Counter

import joblib
import numpy as np
from tqdm import tqdm
from lhotse import CutSet


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cut-file",
        type=str,
    )

    parser.add_argument(
        "--kmeans",
        type=str,
    )

    parser.add_argument(
        "--output-file",
        type=str,
    )
    
    parser.add_argument(
        "--num-workers",
        type=int,
    )

    parser.add_argument("--layer-norm", action="store_true")

    return parser.parse_args()


def normalize(x: np.ndarray, eps=1e-5) -> np.ndarray:
    mu = x.mean(axis=-1, keepdims=True)
    sig = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(sig + eps)


def subsampling(pred, kernel_size, stride):
    new_pred = []
    for i in range(0, len(pred), stride):
        if i + kernel_size <= len(pred): # 卷积无填充，故丢弃不满足 kernel_size 的 pred
            kernel = pred[i:i+kernel_size]
            pred2count = Counter(kernel)
            new_pred.append(pred2count.most_common(1)[0][0]) # 选择 kernel_size 内出现次数最多的 pred 作为 new_pred
    assert len(new_pred) == (len(pred) -(kernel_size - 1) - 1) // stride + 1, print(f"length after subsampling: {len(new_pred)}")
    return new_pred


def simulate_Conv2dSubsampling(pred):
    pred1 = subsampling(pred, 3, 1)
    pred2 = subsampling(pred1, 3, 2)
    pred3 = subsampling(pred2, 3, 1)
    assert len(pred3) == (len(pred) - 7) // 2, print(f"The lengh of subsampled pred: {len(pred3)}")
    return pred3            


def kms_decode(cut, layer_norm):
    fbank = cut.load_features()
    if layer_norm:
        fbank = normalize(fbank)   
    pred = kmeans_model.predict(fbank)
    # simulate Conv2dSubsampling
    # pred = simulate_Conv2dSubsampling(pred)
    pred = ' '.join([str(l) for l in pred])
    return cut, pred


def run(args):
    # decoding
    new_cuts = []
    with ProcessPoolExecutor(args.num_workers) as ex:
        for cut, pred in tqdm(ex.map(kms_decode, cutset, repeat(args.layer_norm)), desc='kmeans decoding'):
            cut.custom= {'kmeans': pred}
            new_cuts.append(cut)
    # save new_cuts
    new_cuts = CutSet.from_cuts(new_cuts)
    new_cuts.to_file(args.output_file)
    

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )
    args = get_args()
    logging.info(str(args))
    # load kmeans model
    kmeans_model = joblib.load(open(args.kmeans, "rb"))
    kmeans_model.verbose = False
    # load cutset
    cutset = CutSet.from_file(args.cut_file)
    run(args)
