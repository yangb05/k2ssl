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


def kms_decode(cut, layer_norm):
    fbank = cut.load_features()
    if layer_norm:
        fbank = normalize(fbank)   
    pred = kmeans_model.predict(fbank)
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