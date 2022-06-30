#!/usr/bin/env python3
import re
import logging
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Sequence
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from copy import deepcopy

from training.config.config import BaseConfig
from training.config.modelconfig import get_model
from training.config.dataconfig import MiaDataConfig, get_dataset, get_dataconfig
from training.data.mia import get_mia_dataset
from training.loss import get_loss
from training.misc_utilities.check_subset import assert_check_subset

from paths import cache_path


def inference(checkpoint, input_paths, outfile, model_key, config = None):
    torch.set_grad_enabled(False)

    if config is None:
        config = BaseConfig(cv_enabled=checkpoint["config"]["cv_enabled"])
        config.cache_path = cache_path
        # check if the loaded config is the same as the codebase
        dataconfig = MiaDataConfig(
            config=config,
            is_validation=True,
            cache_path=config.cache_path,
        )
    else:
        dataconfig = config.dataconfigs['test']
    checkpoint["config"]["dataconfigs"] = None
    assert_check_subset(
        checkpoint["config"],
        config.to_dict(),
        ignore_paths=[
            "config.git_commit",
            "config.created",
            "config._current_fold",
            "config.nickname",
            "config.split",
            "config.dataconfigs",
            "config.MAX_EPOCHS",
            "config.MAX_EPOCHS_LRSCHEDULER",
            "config.modelconfig.decay_lr_until",
            # TODO: ok to skip?
            "config.modelconfig.pretrained_mode",
        ],
    )

    # load all models
    if isinstance(checkpoint[model_key], Sequence):
        all_model_weights = checkpoint[model_key]
    else:
        all_model_weights = [checkpoint[model_key]]

    loss = get_loss(config.modelconfig.loss_fn)

    # create inputs from input_paths
    inputs = []
    for ip in input_paths:
        inputs.append(dict(patient=Path(ip).name, path=str(ip), inf=0, sev=0))
    dataset = get_mia_dataset(dataconfig, labels=pd.DataFrame(inputs))
    loader = DataLoader(
        dataset, batch_size=config.Batch_SIZE, num_workers=config.WORKERS, shuffle=False
    )

    outputs = []

    for model_weights in all_model_weights:
        model = get_model(config)
        model.load_state_dict(model_weights)
        model.to(config.DEVICE)
        model.eval()

        outputs = []
        for sample in tqdm(loader, unit="batch"):
            v_tensor, inf_gt, sev_gt = sample
            output = model(v_tensor.to(config.DEVICE), None, None).cpu()

            outputs.append(deepcopy(output))

    outputs = torch.cat(outputs)
    df = pd.DataFrame(
        outputs.numpy(), columns=[f"out_{i}" for i in range(outputs.size(1))]
    )
    inf_pred, sev_pred = loss.finalize(outputs)
    df["pred_inf"] = inf_pred.numpy()
    df["pred_sev"] = sev_pred.numpy()
    df["patient"] = [inp["patient"] for inp in inputs]
    df["path"] = [inp["path"] for inp in inputs]

    df.to_csv(outfile, index=False)


def cli():
    parser = ArgumentParser()
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("checkpoint")
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--model-key", default="model_ema")

    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    checkpoint = torch.load(args.checkpoint)

    searchdir = Path(args.input)
    inputs = searchdir.glob("**/ct_scan*")

    inference(checkpoint, inputs, args.output, args.model_key)


if __name__ == "__main__":
    cli()
