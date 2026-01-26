import argparse
import logging
import os
import random
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch import nn

import learn2learn as l2l
from learn2learn.data.transforms import (
    FusedNWaysKShots,
    LoadData,
    RemapLabels,
    ConsecutiveLabels,
)

from fault_datasets import CWRU_FFT, CWRU_PhysFeat
from dtn.networks import CNN1dEncoder, LinearClassifier, init_weights
from models import KG_MLP
from utils import setup_logger


SUPPORTED_DATASETS = ["CWRU"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="DTN baseline (train from scratch per episode) on CWRU."
    )

    # few-shot task spec
    parser.add_argument("--ways", type=int, default=10)
    parser.add_argument("--shots", type=int, default=5)

    # DTN inner-loop training (support-set training)
    parser.add_argument(
        "--dtn_steps",
        type=int,
        default=200,
        help="Number of gradient steps on the support set per episode.",
    )
    parser.add_argument("--dtn_lr", type=float, default=1e-3)

    # eval episodes
    parser.add_argument("--episodes", type=int, default=100)

    # device / seed
    parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)

    # dataset
    parser.add_argument("--dataset", type=str, default="CWRU", choices=SUPPORTED_DATASETS)
    parser.add_argument(
        "--preprocess",
        type=str,
        default="FFT",
        help="CWRU preprocess branch: FFT or PHYS (31-D physical features).",
        choices=["FFT", "PHYS"],
    )
    parser.add_argument("--data_dir_path", type=str, default="./data")
    # 你要的“load3 工况”就是 domain=3
    parser.add_argument("--test_domain", type=int, default=3)

    # PHYS / KG related (kept consistent with existing pipeline)
    parser.add_argument("--kg_dir", type=str, default="./data/kg")
    parser.add_argument("--time_steps", type=int, default=1024)
    parser.add_argument("--overlap_ratio", type=float, default=0.5)
    parser.add_argument("--normalization", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fs", type=int, default=12000)
    parser.add_argument("--scale_features", action=argparse.BooleanOptionalAction, default=True)

    # logging
    parser.add_argument("--log", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log_path", type=str, default="./logs")

    return parser.parse_args()


@dataclass
class EpisodeResult:
    loss: float
    acc: float


class DTNFFT(nn.Module):
    """
    DTN FFT branch model (encoder + linear classifier), matching `dtn/networks.py`.
    """

    def __init__(self, ways: int, feature_dim: int = 64):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.ways = int(ways)
        self.encoder = CNN1dEncoder(feature_dim=self.feature_dim, flatten=True)
        self.classifier = LinearClassifier(input_dim=self.feature_dim * 25, num_classes=self.ways)
        init_weights(self.encoder)
        init_weights(self.classifier)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        return self.classifier(feat)


def _split_support_query(
    data: torch.Tensor, labels: torch.Tensor, shots: int, ways: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Match the repo's existing few-shot split convention (see utils.fast_adapt).

    FusedNWaysKShots returns 2*shots examples per class; this split selects
    alternating items as support and the rest as query.
    """
    adaptation_indices = np.zeros(data.size(0), dtype=bool)
    adaptation_indices[np.arange(shots * ways) * 2] = True
    evaluation_indices = torch.from_numpy(~adaptation_indices)
    adaptation_indices = torch.from_numpy(adaptation_indices)

    support_x, support_y = data[adaptation_indices], labels[adaptation_indices]
    query_x, query_y = data[evaluation_indices], labels[evaluation_indices]
    return support_x, support_y, query_x, query_y


def _make_taskset(args):
    if args.preprocess == "FFT":
        base = CWRU_FFT(args.test_domain, args.data_dir_path, fft=True)
    elif args.preprocess == "PHYS":
        base = CWRU_PhysFeat(
            args.test_domain,
            args.data_dir_path,
            time_steps=args.time_steps,
            overlap_ratio=args.overlap_ratio,
            normalization=args.normalization,
            random_seed=args.seed,
            fs=args.fs,
            scale_features=args.scale_features,
        )
    else:
        raise ValueError('Unsupported preprocess. Expected "FFT" or "PHYS".')

    meta = l2l.data.MetaDataset(base)
    transforms = [
        FusedNWaysKShots(meta, n=args.ways, k=2 * args.shots),
        LoadData(meta),
        RemapLabels(meta),
        ConsecutiveLabels(meta),
    ]
    return l2l.data.Taskset(meta, task_transforms=transforms, num_tasks=args.episodes)


def _make_model(args, device):
    if args.preprocess == "FFT":
        # DTN: train from scratch per episode (no pretraining).
        model = DTNFFT(ways=args.ways, feature_dim=64).to(device)
    elif args.preprocess == "PHYS":
        # For PHYS branch we still need a KG file to instantiate KG_MLP (its buffers),
        # but DTN still trains from scratch per episode.
        kg_path = os.path.join(args.kg_dir, f"kg_domain{int(args.test_domain)}_W_P.npz")
        if not os.path.exists(kg_path):
            # fallback to first existing KG file
            for cand in [0, 1, 2, 3]:
                p = os.path.join(args.kg_dir, f"kg_domain{int(cand)}_W_P.npz")
                if os.path.exists(p):
                    kg_path = p
                    break
        model = KG_MLP.from_kg_file(output_size=args.ways, kg_npz_path=kg_path).to(device)
    else:
        raise ValueError('Unsupported preprocess. Expected "FFT" or "PHYS".')
    return model


def _episode_train_eval(args, taskset, device) -> EpisodeResult:
    model = _make_model(args, device)
    model.train()

    batch = taskset.sample()
    data, labels = batch
    data, labels = data.to(device), labels.to(device)
    support_x, support_y, query_x, query_y = _split_support_query(data, labels, args.shots, args.ways)

    opt = torch.optim.Adam(model.parameters(), lr=args.dtn_lr)
    loss_fn = nn.CrossEntropyLoss(reduction="mean")

    for _ in range(int(args.dtn_steps)):
        opt.zero_grad()
        logits = model(support_x)
        loss = loss_fn(logits, support_y)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        logits_q = model(query_x)
        loss_q = loss_fn(logits_q, query_y).item()
        pred = torch.argmax(logits_q, dim=1)
        acc = (pred == query_y).float().mean().item()
    return EpisodeResult(loss=loss_q, acc=acc)


def main():
    args = parse_args()
    if args.dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"Dataset must be one of {SUPPORTED_DATASETS}.")
    if args.preprocess not in ["FFT", "PHYS"]:
        raise ValueError('Preprocess must be "FFT" or "PHYS".')

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # device
    if args.cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        device = torch.device("cuda")
        logging.info("Training DTN with CUDA.")
    else:
        device = torch.device("cpu")
        logging.info("Training DTN with CPU.")

    experiment_title = f"DTN_{args.dataset}_{args.preprocess}_{args.ways}w{args.shots}s_load{int(args.test_domain)}"
    if args.log:
        if not os.path.exists(args.log_path):
            os.makedirs(args.log_path)
        setup_logger(args.log_path, experiment_title)

    logging.info(f"Experiment: {experiment_title}")
    logging.info(f"DTN on CWRU domain(load)={int(args.test_domain)} | episodes={int(args.episodes)}")
    logging.info(f"Task: {int(args.ways)}-way {int(args.shots)}-shot | dtn_steps={int(args.dtn_steps)} | lr={float(args.dtn_lr)}")

    taskset = _make_taskset(args)
    accs = []
    losses = []

    for ep in range(int(args.episodes)):
        r = _episode_train_eval(args, taskset, device)
        accs.append(r.acc)
        losses.append(r.loss)
        if (ep + 1) % max(1, int(args.episodes) // 5) == 0:
            logging.info(f"Episode {ep+1}/{int(args.episodes)} | acc={r.acc:.4f} | loss={r.loss:.4f}")

    accs = np.asarray(accs, dtype=np.float32)
    losses = np.asarray(losses, dtype=np.float32)
    logging.info("==== DTN Summary (per-episode query set) ====")
    logging.info(f"Accuracy: mean={accs.mean():.4f}, std={accs.std(ddof=1):.4f}")
    logging.info(f"Loss:     mean={losses.mean():.4f}, std={losses.std(ddof=1):.4f}")

    print(
        f"[DTN] load{int(args.test_domain)} | "
        f"acc mean={accs.mean():.4f} std={accs.std(ddof=1):.4f} | "
        f"loss mean={losses.mean():.4f}"
    )


if __name__ == "__main__":
    main()

