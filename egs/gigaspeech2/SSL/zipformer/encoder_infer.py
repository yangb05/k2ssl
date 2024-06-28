#!/usr/bin/env python3
# Copyright    2021-2024  Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                       Wei Kang,
#                                                       Mingshuang Luo,
#                                                       Zengwei Yao,
#                                                       Yifan Yang,
#                                                       Daniel Povey)
#
# Copyright    2024  Shanghai Jiao Tong University  (authors: Jianheng Zhuo)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Usage:

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

# For HuBERT model finetuning:
./hubert/finetune.py \
  --world-size 8 \
  --num-epochs 200 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir hubert/exp \
  --full-libri 0 \
  --max-duration 1000

It supports finetuning with:
  - transducer loss (default), with `--use-transducer True --use-ctc False`
  - ctc loss (not recommended), with `--use-transducer False --use-ctc True`
  - transducer loss & ctc loss, with `--use-transducer True --use-ctc True`
"""


import sys
import random
import argparse
import copy
import logging
import warnings
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, Optional, Tuple, Union

import k2
from tqdm import tqdm
import optim
import sentencepiece as spm
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from ssl_datamodule import VietnameseDataModule
from decoder import Decoder
from hubert_ce import HubertModel
from joiner import Joiner
from lhotse.cut import Cut
from lhotse.dataset.sampling.base import CutSampler
from lhotse.utils import fix_random_seed
from model import AsrModel
from optim import Eden, ScaledAdam
from torch import Tensor
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from icefall import diagnostics
from icefall.checkpoint import load_checkpoint, remove_checkpoints
from icefall.checkpoint import save_checkpoint as save_checkpoint_impl
from icefall.checkpoint import (
    save_checkpoint_with_global_batch_idx,
    update_averaged_model,
)
from icefall.dist import cleanup_dist, setup_dist
from icefall.env import get_env_info
from icefall.hooks import register_inf_check_hooks
from icefall.utils import (
    AttributeDict,
    MetricsTracker,
    get_parameter_groups_with_lrs,
    setup_logger,
    str2bool,
    make_pad_mask,
)

LRSchedulerType = Union[torch.optim.lr_scheduler._LRScheduler, optim.LRScheduler]


def get_adjusted_batch_count(params: AttributeDict) -> float:
    # returns the number of batches we would have used so far if we had used the reference
    # duration.  This is for purposes of set_batch_count().
    return (
        params.batch_idx_train
        * params.accum_grad
        * (params.max_duration * params.world_size)
        / params.ref_duration
    )


def set_batch_count(model: Union[nn.Module, DDP], batch_count: float) -> None:
    if isinstance(model, DDP):
        # get underlying nn.Module
        model = model.module
    for name, module in model.named_modules():
        if hasattr(module, "batch_count"):
            module.batch_count = batch_count
        if hasattr(module, "name"):
            module.name = name


def add_model_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--num-encoder-layers",
        type=str,
        default="2,2,3,4,3,2",
        help="Number of zipformer encoder layers per stack, comma separated.",
    )

    parser.add_argument(
        "--downsampling-factor",
        type=str,
        default="1,2,4,8,4,2",
        help="Downsampling factor for each stack of encoder layers.",
    )

    parser.add_argument(
        "--feedforward-dim",
        type=str,
        default="512,768,1024,1536,1024,768",
        help="Feedforward dimension of the zipformer encoder layers, per stack, comma separated.",
    )

    parser.add_argument(
        "--num-heads",
        type=str,
        default="4,4,4,8,4,4",
        help="Number of attention heads in the zipformer encoder layers: a single int or comma-separated list.",
    )

    parser.add_argument(
        "--feature-dim",
        type=int,
        default=80,
        help="Dim of fbank feature.",
    )
    
    parser.add_argument(
        "--encoder-dim",
        type=str,
        default="192,256,384,512,384,256",
        help="Embedding dimension in encoder stacks: a single int or comma-separated list.",
    )

    parser.add_argument(
        "--query-head-dim",
        type=str,
        default="32",
        help="Query/key dimension per head in encoder stacks: a single int or comma-separated list.",
    )

    parser.add_argument(
        "--value-head-dim",
        type=str,
        default="12",
        help="Value dimension per head in encoder stacks: a single int or comma-separated list.",
    )

    parser.add_argument(
        "--pos-head-dim",
        type=str,
        default="4",
        help="Positional-encoding dimension per head in encoder stacks: a single int or comma-separated list.",
    )

    parser.add_argument(
        "--pos-dim",
        type=int,
        default="48",
        help="Positional-encoding embedding dimension",
    )

    parser.add_argument(
        "--encoder-unmasked-dim",
        type=str,
        default="192,192,256,256,256,192",
        help="Unmasked dimensions in the encoders, relates to augmentation during training.  "
        "A single int or comma-separated list.  Must be <= each corresponding encoder_dim.",
    )

    parser.add_argument(
        "--cnn-module-kernel",
        type=str,
        default="31,31,15,15,15,31",
        help="Sizes of convolutional kernels in convolution modules in each encoder stack: "
        "a single int or comma-separated list.",
    )

    # hubert parameters
    parser.add_argument(
        "--label-rate",
        type=float,
        default=100,
    )

    parser.add_argument(
        "--sample-rate",
        type=float,
        default=100,
    )

    parser.add_argument(
        "--extractor-mode",
        type=str,
        default="default",
        help="""mode for feature extractor, should in EXTRACTOR_MODE_CHOICES. default has a single group 
            norm with d groups in the first conv block, whereas layer_norm 
            has layer norms in every block (meant to use with normalize=True)""",
    )

    parser.add_argument(
        "--conv-feature-layers",
        type=str,
        default="[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2",
        help="string describing convolutional feature extraction layers in form of a python list that contains [(dim, kernel_size, stride), ...]",
    )

    parser.add_argument(
        "--conv-bias", type=bool, default=False, help="include bias in conv encoder"
    )

    parser.add_argument(
        "--feature-grad-mult",
        type=float,
        default=1.0,
        help="multiply feature extractor var grads by this",
    )

    # masking
    parser.add_argument("--mask-length", type=int, default=10, help="mask_length")

    parser.add_argument(
        "--mask-prob",
        type=float,
        default=0.65,
        help="probability of replacing a token with mask",
    )

    parser.add_argument(
        "--mask-selection",
        type=str,
        choices=["static", "uniform", "normal", "poisson"],
        default="static",
        help="how to choose mask length",
    )

    parser.add_argument(
        "--mask-other",
        type=float,
        default=0,
        help="secondary mask argument (used for more complex distributions),see help in compute_mask_indicesh",
    )

    parser.add_argument(
        "--no-mask-overlap",
        type=bool,
        default=False,
        help="whether to allow masks to overlap",
    )

    parser.add_argument(
        "--mask-min-space",
        type=int,
        default=1,
        help="min space between spans (if no overlap is enabled)",
    )

    # channel masking
    parser.add_argument(
        "--mask-channel-length",
        type=int,
        default=10,
        help="length of the mask for features (channels)",
    )

    parser.add_argument(
        "--mask-channel-prob",
        type=float,
        default=0.0,
        help="probability of replacing a feature with 0",
    )

    parser.add_argument(
        "--mask-channel-selection",
        type=str,
        choices=["static", "uniform", "normal", "poisson"],
        default="static",
        help="how to choose mask length for channel masking",
    )

    parser.add_argument(
        "--mask-channel-other",
        type=float,
        default=0,
        help="secondary mask argument (used for more complex distributions), see help in compute_mask_indicesh",
    )

    parser.add_argument(
        "--no-mask-channel-overlap",
        type=bool,
        default=False,
        help="whether to allow channel masks to overlap",
    )

    parser.add_argument(
        "--mask-channel-min-space",
        type=int,
        default=1,
        help="min space between spans (if no overlap is enabled)",
    )

    # loss computation
    parser.add_argument(
        "--skip-masked",
        type=bool,
        default=False,
        help="skip computing losses over masked frames",
    )

    parser.add_argument(
        "--skip-nomask",
        type=bool,
        default=False,
        help="skip computing losses over unmasked frames",
    )

    parser.add_argument(
        "--checkpoint-activations",
        type=bool,
        default=False,
        help="recompute activations and save memory for extra compute",
    )

    parser.add_argument(
        "--pred-masked-weight",
        type=float,
        default=1,
        help="weight for masked part in ssl loss",
    )

    parser.add_argument(
        "--pred-nomask-weight",
        type=float,
        default=0,
        help="weight for masked part in ssl loss",
    )

    parser.add_argument(
        "--loss-weights",
        type=float,
        nargs="*",
        default=[10],
        help="weight for masked part in ssl loss",
    )

    # FP16 optimization
    parser.add_argument(
        "--required-seq-len-multiple",
        type=int,
        default=2,
        help="pad the input to encoder such that the sequence length is divisible by multiple",
    )

    parser.add_argument(
        "--attn-type", type=str, default="", help="if espnet use ESPNET MHA"
    )

    parser.add_argument(
        "--pos-enc-type",
        type=str,
        default="abs",
        help="Positional encoding type to use in conformer",
    )

    parser.add_argument(
        "--logit-temp", type=float, default=0.1, help="temperature to divide logits by"
    )

    parser.add_argument(
        "--dropout-input",
        type=float,
        default=0.0,
        help="dropout to apply to the input (after feat extr)",
    )

    parser.add_argument(
        "--dropout-features",
        type=float,
        default=0.0,
        help="dropout to apply to the features (after feat extr)",
    )

    parser.add_argument(
        "--num-classes",
        type=int,
        nargs="*",
        default=[504],
        help="""num class, a little larger than the number of cluster,
        the largest is for padding, 
        and the value should be the multiple of 4, for faster computation""",
    )

    parser.add_argument(
        "--untie-final-proj",
        type=bool,
        default=False,
        help="use separate projection for each target",
    )

    parser.add_argument(
        "--decoder-dim",
        type=int,
        default=512,
        help="Embedding dimension in the decoder model.",
    )

    parser.add_argument(
        "--joiner-dim",
        type=int,
        default=512,
        help="""Dimension used in the joiner model.
        Outputs from the encoder and decoder model are projected
        to this dimension before adding.
        """,
    )

    parser.add_argument(
        "--use-transducer",
        type=str2bool,
        default=True,
        help="If True, use Transducer head.",
    )

    parser.add_argument(
        "--use-ctc",
        type=str2bool,
        default=False,
        help="If True, use CTC head.",
    )


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Number of GPUs for DDP training.",
    )

    parser.add_argument(
        "--master-port",
        type=int,
        default=12354,
        help="Master port to use for DDP training.",
    )

    parser.add_argument(
        "--tensorboard",
        type=str2bool,
        default=True,
        help="Should various information be logged in tensorboard.",
    )

    parser.add_argument(
        "--num-epochs",
        type=int,
        default=400,
        help="Number of epochs to train.",
    )

    parser.add_argument(
        "--start-epoch",
        type=int,
        default=1,
        help="""Resume training from this epoch. It should be positive.
        If larger than 1, it will load checkpoint from
        exp-dir/epoch-{start_epoch-1}.pt
        """,
    )

    parser.add_argument(
        "--start-batch",
        type=int,
        default=0,
        help="""If positive, --start-epoch is ignored and
        it loads the checkpoint from exp-dir/checkpoint-{start_batch}.pt
        """,
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="zipformer/exp",
        help="""The experiment dir.
        It specifies the directory where all training related
        files, e.g., checkpoints, log, etc, are saved
        """,
    )

    parser.add_argument(
        "--base-lr", type=float, default=0.045, help="The base learning rate."
    )
    
    parser.add_argument(
        "--bpe-model",
        type=str,
        default="data/lang_bpe_2000/bpe.model",
        help="Path to the BPE model",
    )
    
    parser.add_argument(
        "--pretrained-dir",
        type=str,
        default="zipformer/exp_pretrain_vietnamese_mask_0.8_label_fixed/best-valid-loss.pt",
        help="""The pretrained model dir.
        It specifies the directory where the pretrained checkpoint is saved.""",
    )

    parser.add_argument(
        "--lr-batches",
        type=float,
        default=7500,
        help="""Number of steps that affects how rapidly the learning rate
        decreases. We suggest not to change this.""",
    )

    parser.add_argument(
        "--lr-epochs",
        type=float,
        default=10.5,
        help="""Number of epochs that affects how rapidly the learning rate decreases.
        """,
    )

    parser.add_argument(
        "--warmup-batches",
        type=float,
        default=5000,
        help="Eden warmup steps",
    )

    parser.add_argument(
        "--warmup-start",
        type=float,
        default=0,
        help="Eden warmup start learning rate",
    )

    parser.add_argument(
        "--ref-duration",
        type=float,
        default=600,
        help="Reference batch duration for purposes of adjusting batch counts for setting various "
        "schedules inside the model",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed for random generators intended for reproducibility",
    )

    parser.add_argument(
        "--print-diagnostics",
        type=str2bool,
        default=False,
        help="Accumulate stats on activations, print them and exit.",
    )

    parser.add_argument(
        "--sanity-check",
        type=str2bool,
        default=False,
        help="Check if any of the batches in epoch 1 would cause OOM.",
    )

    parser.add_argument(
        "--inf-check",
        type=str2bool,
        default=False,
        help="Add hooks to check for infinite module outputs and gradients.",
    )

    parser.add_argument(
        "--save-every-n",
        type=int,
        default=100000,
        help="""Save checkpoint after processing this number of batches"
        periodically. We save checkpoint to exp-dir/ whenever
        params.batch_idx_train % save_every_n == 0. The checkpoint filename
        has the form: f'exp-dir/checkpoint-{params.batch_idx_train}.pt'
        Note: It also saves checkpoint to `exp-dir/epoch-xxx.pt` at the
        end of each epoch where `xxx` is the epoch number counting from 1.
        """,
    )

    parser.add_argument(
        "--keep-last-k",
        type=int,
        default=30,
        help="""Only keep this number of checkpoints on disk.
        For instance, if it is 3, there are only 3 checkpoints
        in the exp-dir with filenames `checkpoint-xxx.pt`.
        It does not affect checkpoints with name `epoch-xxx.pt`.
        """,
    )

    parser.add_argument(
        "--average-period",
        type=int,
        default=200,
        help="""Update the averaged model, namely `model_avg`, after processing
        this number of batches. `model_avg` is a separate version of model,
        in which each floating-point parameter is the average of all the
        parameters from the start of training. Each time we take the average,
        we do: `model_avg = model * (average_period / batch_idx_train) +
            model_avg * ((batch_idx_train - average_period) / batch_idx_train)`.
        """,
    )

    parser.add_argument(
        "--accum-grad",
        type=int,
        default=4,
        help="""update gradient when batch_idx_train % accum_grad == 0.
        """,
    )

    parser.add_argument(
        "--use-fp16",
        type=str2bool,
        default=False,
        help="Whether to use half precision training.",
    )

    parser.add_argument(
        "--max-keep-size",
        type=int,
        default=sys.maxsize,
        help="exclude sample longer than this.",
    )

    parser.add_argument(
        "--min-keep-size",
        type=float,
        default=200,
        help="exclude sample less than this.",
    )

    parser.add_argument(
        "--max-sample-size",
        type=float,
        default=250000,
        help="max sample size to crop to for batching.",
    )

    add_model_arguments(parser)

    return parser


def get_params() -> AttributeDict:
    """Return a dict containing training parameters.

    All training related parameters that are not passed from the commandline
    are saved in the variable `params`.

    Commandline options are merged into `params` after they are parsed, so
    you can also access them via `params`.

    Explanation of options saved in `params`:

        - best_train_loss: Best training loss so far. It is used to select
                           the model that has the lowest training loss. It is
                           updated during the training.

        - best_valid_loss: Best validation loss so far. It is used to select
                           the model that has the lowest validation loss. It is
                           updated during the training.

        - best_train_epoch: It is the epoch that has the best training loss.

        - best_valid_epoch: It is the epoch that has the best validation loss.

        - batch_idx_train: Used to writing statistics to tensorboard. It

                           contains number of updates happen to the model so far across
                           epochs.

        - sub_batch_idx_train: It contains number of batch trained so far across
                           epochs.

        - log_interval:  Print training loss if batch_idx % log_interval` is 0

        - reset_interval: Reset statistics if batch_idx % reset_interval is 0

        - valid_interval:  Run validation if batch_idx % valid_interval is 0

        - warm_step: The warmup period that dictates the decay of the
              scale on "simple" (un-pruned) loss.
    """
    params = AttributeDict(
        {
            "best_train_loss": float("inf"),
            "best_valid_loss": float("inf"),
            "best_train_epoch": -1,
            "best_valid_epoch": -1,
            "batch_idx_train": 0,
            "sub_batch_idx_train": 0,
            "log_interval": 50,
            "reset_interval": 200,
            "valid_interval": 3000,  # For the 100h subset, use 800
            # parameters for pruned RNN-T loss
            "warm_step": 2000,
            "env_info": get_env_info(),
        }
    )

    return params


def _to_int_tuple(s: str):
    return tuple(map(int, s.split(",")))


def get_encoder_model(params: AttributeDict) -> nn.Module:
    if hasattr(params, "pretrained_dir"):
        logging.info(f"Loading {params.pretrained_dir}")
        pretrained = torch.load(params.pretrained_dir)
        encoder = HubertModel(params)
        encoder.load_state_dict(pretrained["model"])
    else:
        encoder = HubertModel(params)
    return encoder


def compute_hidden_states(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    batch: dict,
    is_training: bool,
    results: dict
) -> Tuple[Tensor, MetricsTracker]:
    """
    Compute hidden states given the model and its inputs.

    Args:
      params:
        Parameters for training. See :func:`get_params`.
      model:
        The model for training. It is an instance of Zipformer in our case.
      batch:
        A batch of data. See `dataset.HubertDataset()`
        for the content in it.
      is_training:
        True for training. False for validation. When it is True, this
        function enables autograd during computation; when it is False, it
        disables autograd.
    """
    device = model.device if isinstance(model, DDP) else next(model.parameters()).device
    features = batch["features"].to(device)
    # at entry, features is (N, T, C)
    assert features.ndim == 3

    padding_mask = batch["padding_mask"].to(device)
    kmeans = batch["kmeans"].to(device)

    with torch.set_grad_enabled(is_training):
        outputs = model(
            source=features, target_list=[kmeans], padding_mask=padding_mask, mask=False, features_only=True
        )
    hidden_states = outputs['x']
    for idx, cut in enumerate(batch['cuts']):
        key = cut.id
        value = hidden_states[idx].detach().cpu().numpy()
        results.update({key: value})
    return results

  
def infer(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    valid_dl: torch.utils.data.DataLoader
) -> Dict:
    """Run the validation process."""
    model.eval()
    results = {}
    for batch in tqdm(valid_dl, desc="infering"):
        results = compute_hidden_states(
            params=params,
            model=model,
            batch=batch,
            is_training=False,
            results = results
        )
    return results


def run(args):
    """
    Args:
      args:
        The return value of get_parser().parse_args()
    """
    params = get_params()
    params.update(vars(args))

    fix_random_seed(params.seed)

    setup_logger(f"{params.exp_dir}/log/log-infer")
    logging.info("Infering started")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    logging.info(f"Device: {device}")
    logging.info(params)
    logging.info("About to create encoder")
    encoder = get_encoder_model(params)
    encoder.to(device)

    vietnamese = VietnameseDataModule(args)
    train_cuts = vietnamese.train_cuts()
    # sample 1/10 cuts
    num_cuts = len(train_cuts) // 10
    random.seed(params.seed)
    train_cuts = train_cuts.shuffle(rng=random).subset(first=num_cuts)
    sampler_state_dict = None
    train_dl = vietnamese.train_dataloaders(
        train_cuts,
        max_sample_size=params.max_sample_size,
        sample_rate=params.sample_rate,
        label_rate=params.label_rate,
        num_classes=params.num_classes,
        sampler_state_dict=sampler_state_dict,
    )
    train_results = infer(params, encoder, train_dl)
    
    np.save('/data_a100/userhome/yangb/data/hidden_states/viet_220h.npy', train_results)
    logging.info("Done!")


def main():
    parser = get_parser()
    VietnameseDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)
    run(args=args)


torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == "__main__":
    main()
