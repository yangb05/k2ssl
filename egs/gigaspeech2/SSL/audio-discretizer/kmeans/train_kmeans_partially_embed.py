#!/usr/bin/env python3

import argparse
import logging
import random
import math
import os
default_n_threads = 8
os.environ['OPENBLAS_NUM_THREADS'] = f"{default_n_threads}"
os.environ['MKL_NUM_THREADS'] = f"{default_n_threads}"
os.environ['OMP_NUM_THREADS'] = f"{default_n_threads}"

import joblib
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
from lhotse import CutSet



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed-files", type=str)
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
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--layer-norm", action="store_true")
    args = parser.parse_args()
    return args


# apply layer norm on the last dim
def layer_norm(x: np.ndarray, eps=1e-5) -> np.ndarray:
    mu = x.mean(axis=-1, keepdims=True)
    sig = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(sig + eps)


def load_embed(embed_files):
    cut_embeds = {}
    for file in embed_files.split(','):
        cut_embeds.update(np.load(file, allow_pickle=True).item())
    return cut_embeds


def load_embeds(embed_files, part_size, percent=1.0, apply_layer_norm=False, seed=42):
    # get embeds
    cut_embeds = load_embed(embed_files)
    cut_ids = list(cut_embeds.keys())
    random.seed(seed)
    random.shuffle(cut_ids)
    sampled_len = int(len(cut_ids) * percent)
    sampled_ids = cut_ids[:sampled_len]
    for i in tqdm(range(0, sampled_len, part_size)):
        part_ids = sampled_ids[i : min(i + part_size, sampled_len)]
        part_embeds = []
        for i in tqdm(part_ids):
            embed = cut_embeds[i]
            if apply_layer_norm:
                embed = layer_norm(embed)
            part_embeds.append(embed)
        part_embeds = np.concatenate(part_embeds, axis=0)
        yield part_embeds


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
    for _ in range(args.epochs):
        for part_embeds in load_embeds(
            args.embed_files, args.batch_size, args.percent, args.layer_norm, args.seed
        ):
            model.partial_fit(part_embeds)
            inertia = -model.score(part_embeds) / len(part_embeds)
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
