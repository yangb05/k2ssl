#!/usr/bin/env python3

import argparse
import logging
import random
import os
default_n_threads = 20
os.environ['OPENBLAS_NUM_THREADS'] = f"{default_n_threads}"
os.environ['MKL_NUM_THREADS'] = f"{default_n_threads}"
os.environ['OMP_NUM_THREADS'] = f"{default_n_threads}"

import joblib
import kaldiio
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
from lhotse import CutSet


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cut-file", type=str)
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--n-clusters", type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--percent", default=1, type=float)
    parser.add_argument("--init", default="k-means++", type=str)
    parser.add_argument("--max-iter", default=100, type=int)
    parser.add_argument("--batch-size", default=10000, type=int)
    parser.add_argument("--tol", default=0.0, type=float)
    parser.add_argument("--max-no-improvement", default=100, type=int)
    parser.add_argument("--n-init", default=20, type=int)
    parser.add_argument("--reassignment-ratio", default=0.0, type=float)
    parser.add_argument("--layer-norm", action="store_true")
    args = parser.parse_args()
    return args


# apply layer norm on the last dim
def layer_norm(x: np.ndarray, eps=1e-5) -> np.ndarray:
    mu = x.mean(axis=-1, keepdims=True)
    sig = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(sig + eps)


def load_feats(cut_file, part_size, max_iter, percent=1.0, apply_layer_norm=False, seed=42):
    random.seed(seed)
    cutset = CutSet.from_file(cut_file)
    if percent > 0 and percent < 1:
        sampled_len = int(len(cutset) * percent)
        cutset = cutset.sample(n_cuts=sampled_len)
    chunks = cutset.split(num_splits=len(cutset)//part_size)
    for chunk in chunks:
        part_feats = []
        for cut in chunk:
            feat = cut.load_features()
            if apply_layer_norm:
                feat = layer_norm(feat)
            part_feats.append(feat)
        part_feats = np.concatenate(part_feats, axis=0)
        yield part_feats


def main(args):
    model = MiniBatchKMeans(
        n_clusters=args.n_clusters,
        init=args.init,
        max_iter=args.max_iter,
        batch_size=args.batch_size,
        verbose=1,
        compute_labels=False,
        tol=args.tol,
        max_no_improvement=args.max_no_improvement,
        init_size=None,
        n_init=args.n_init,
        reassignment_ratio=args.reassignment_ratio,
    )

    for part_feats in load_feats(
        args.cut_file, args.batch_size, args.max_iter, args.percent, args.layer_norm, args.seed
    ):
        model.partial_fit(part_feats)
        inertia = -model.score(part_feats) / len(part_feats)
        logging.info(f"Total inertia: {inertia:.5f}")

    joblib.dump(model, args.model_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )
    args = get_args()
    logging.info(str(args))
    main(args)
