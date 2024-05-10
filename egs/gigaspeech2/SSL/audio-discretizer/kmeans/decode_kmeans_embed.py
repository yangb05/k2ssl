import argparse
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

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
        "--embed-file",
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


def layer_norm(x: np.ndarray, eps=1e-5) -> np.ndarray:
    mu = x.mean(axis=-1, keepdims=True)
    sig = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(sig + eps)


def kms_decode(cut):
    embed = cut_embeds[cut.id] 
    pred = kmeans_model.predict(embed)
    pred = ' '.join([str(l) for l in pred])
    return cut, pred


def run(args):
    # decoding
    new_cuts = []
    with ProcessPoolExecutor(args.num_workers) as ex:
        for cut, pred in tqdm(ex.map(kms_decode, cutset), desc='kmeans decoding'):
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
    # load embed
    cut_embeds = np.load(args.embed_file, allow_pickle=True).item()
    run(args)
