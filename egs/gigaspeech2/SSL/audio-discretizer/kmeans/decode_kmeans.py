import argparse
import logging
import math
from pathlib import Path

import joblib
import numpy as np
from kaldiio import WriteHelper, load_mat
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--feats-scp",
        type=Path,
    )

    parser.add_argument(
        "--kmeans",
        type=Path,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
    )

    parser.add_argument("--layer-norm", action="store_true")

    return parser.parse_args()


def layer_norm(x: np.ndarray, eps=1e-5) -> np.ndarray:
    mu = x.mean(axis=-1, keepdims=True)
    sig = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(sig + eps)


def run(args):
    kmeans_model = joblib.load(open(args.kmeans, "rb"))
    kmeans_model.verbose = False

    # load feats.scp
    utt2feat = {}
    with open(args.feats_scp, "r") as f:
        utt2feat = dict([l.strip().split(maxsplit=1) for l in f.readlines()])
    uttids = list(utt2feat.keys())

    # decoding
    wspec = f"ark,scp:{args.output_dir}/feats.ark,{args.output_dir}/feats.scp"
    logging.info(f"About to write pred to {wspec}")
    with WriteHelper(wspec) as writer:
        for uttid in tqdm(uttids):
            try:
                feat = load_mat(utt2feat[uttid])
                if args.layer_norm:
                    feat = layer_norm(feat)
                pred = kmeans_model.predict(feat)
                pred = np.expand_dims(pred, axis=1).astype(np.float32)
            except:
                logging.info(f"Error when processing {uttid}")

            writer[uttid] = pred


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )
    args = get_args()
    logging.info(str(args))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run(args)


if __name__ == "__main__":
    main()
