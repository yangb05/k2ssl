#!/usr/bin/env python3

import argparse
import random

import joblib
import kaldiio
import numpy as np
from sklearn.cluster import MiniBatchKMeans


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("feat_scp", type=str)
    parser.add_argument("model_path", type=str)
    parser.add_argument("--n_clusters", type=int, default=128)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--percent", default=1, type=float, help="sample a subset; 1 for all"
    )
    parser.add_argument("--init", default="k-means++")
    parser.add_argument("--max_iter", default=100, type=int)
    parser.add_argument("--batch_size", default=10000, type=int)
    parser.add_argument("--tol", default=0.0, type=float)
    parser.add_argument("--max_no_improvement", default=100, type=int)
    parser.add_argument("--n_init", default=20, type=int)
    parser.add_argument("--reassignment_ratio", default=0.0, type=float)
    # use layer norm as mentioned in AudioLM
    parser.add_argument("--layer-norm", action="store_true")
    args = parser.parse_args()
    return args


# apply layer norm on the last dim
def layer_norm(x: np.ndarray, eps=1e-5) -> np.ndarray:
    mu = x.mean(axis=-1, keepdims=True)
    sig = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(sig + eps)


def load_feats(feat_scp, percent=1.0, apply_layer_norm=False, seed=42):
    utt2feat = {}
    with open(feat_scp, "r") as f:
        utt2feat = dict([l.strip().split(maxsplit=1) for l in f.readlines()])
    uttids = list(utt2feat.keys())
    # if percent<1, shuffle and take percent
    if percent > 0 and percent < 1:
        random.seed(seed)
        random.shuffle(uttids)
        n_samples = int(len(uttids) * percent)
        sampled_uttids = uttids[:n_samples]
    else:
        sampled_uttids = uttids
        n_samples = len(uttids)
    sampled_feats = []
    for uttid in sampled_uttids:
        feat = kaldiio.load_mat(utt2feat[uttid])
        # apply layer norm
        if apply_layer_norm:
            feat = layer_norm(feat)
        sampled_feats.append(feat)
    sampled_feats = np.concatenate(sampled_feats, axis=0)
    print(f"sampled {n_samples} utterances, {len(sampled_feats)} frames.")
    return sampled_feats


def main(args):
    sampled_feats = load_feats(args.feat_scp, args.percent, args.layer_norm, args.seed)
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
    model.fit(sampled_feats)
    joblib.dump(model, args.model_path)
    inertia = -model.score(sampled_feats) / len(sampled_feats)
    print("total inertia: %.5f", inertia)
    print("finished successfully")


if __name__ == "__main__":
    args = get_args()
    print(str(args))
    main(args)
