import logging
from pathlib import Path
import shutil
import torch
from torch.utils.tensorboard import SummaryWriter as TorchSummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard.summary import hparams
import pandas as pd

from training.config.config import BaseConfig


# https://github.com/pytorch/pytorch/issues/32651#issuecomment-648340103
class SummaryWriter(TorchSummaryWriter):
    def add_hparams(
        self, hparam_dict, metric_dict, hparam_domain_discrete=None, run_name=None
    ):
        torch._C._log_api_usage_once(
            "tensorboard.logging.add_hparams"
        )  # pylint: disable=protected-access
        if not isinstance(hparam_dict, dict) or not isinstance(metric_dict, dict):
            raise TypeError("hparam_dict and metric_dict should be dictionary.")
        exp, ssi, sei = hparams(hparam_dict, metric_dict, hparam_domain_discrete)

        self.file_writer.add_summary(exp)
        self.file_writer.add_summary(ssi)
        self.file_writer.add_summary(sei)
        for k, v in metric_dict.items():
            self.add_scalar(k, v)

    def add_hparams2(self, hparam_dict, metric_dict):
        exp, ssi, sei = hparams(hparam_dict, metric_dict, None)
        self.file_writer.add_summary(exp)
        self.file_writer.add_summary(ssi)
        self.file_writer.add_summary(sei)


def tensorboard_dir_to_df(tensorboard_dir: Path) -> pd.DataFrame:
    tf_files = sorted(tensorboard_dir.glob("events.out.tfevents.*"))
    frames = []
    for path in tf_files:
        assert path.is_file()
        df = tensorboard2df(path)
        df["tf_name"] = path.name
        frames.append(df)
    df = pd.concat(frames)
    if not (df.groupby(["tag", "step"]).size() == 1).all():
        raise NotImplementedError("Duplicate tensorboard files are not supported")
    return df


def tensorboard2df(event_file: str) -> pd.DataFrame:
    event_file: Path = Path(event_file)
    if event_file.is_dir():
        logging.warn("Use tensorboard_dir_to_df instead")
        event_file = next(event_file.glob("events.out.tfevents.*"))

    accumulator = EventAccumulator(str(event_file))
    accumulator.Reload()
    rows = []
    for tag in accumulator.scalars.Keys():
        for event in accumulator.scalars.Items(tag):
            rows.append(
                {
                    "tag": tag,
                    "time": event.wall_time,
                    "step": event.step,
                    "value": event.value,
                }
            )
    df = pd.DataFrame(rows)
    return df


def collect_fold_tensorboards(cv_root: Path, num_folds: int):
    frames = []
    for cvdir in cv_root.iterdir():
        if cvdir.name.startswith("cv") and cvdir.is_dir():
            df = tensorboard_dir_to_df(cvdir)
            df["fold"] = cvdir.name
            frames.append(df)
    df: pd.DataFrame = pd.concat(frames)

    # remove all steps that are not complete in all folds
    group_sizes = df.groupby(["tag", "step"]).size()
    df = (
        df.set_index(["tag", "step"])
        .loc[group_sizes[group_sizes == num_folds].index]
        .reset_index()
    )

    return df


def log_cv_tensorboard_summary(cv_root: Path, num_folds: int, name="summary"):
    summary_dir = cv_root / name
    if summary_dir.exists():
        shutil.rmtree(summary_dir)

    df = collect_fold_tensorboards(cv_root, num_folds)
    if len(df) == 0:
        return

    df: pd.DataFrame = (
        df.groupby(["tag", "step"])["value"].mean().reset_index().sort_values("step")
    )
    if len(df) == 0:
        return

    summary_dir.mkdir(exist_ok=False)

    with SummaryWriter(summary_dir) as writer:
        for _, row in df.iterrows():
            writer.add_scalar(row.tag, scalar_value=row.value, global_step=row.step)


def log_cv_tensorboard_bestsummary(
    cv_root: Path, num_folds: int, config: BaseConfig, name="best"
):
    summary_dir = cv_root / name
    if summary_dir.exists():
        shutil.rmtree(summary_dir)

    df = collect_fold_tensorboards(cv_root, num_folds)
    if len(df) == 0:
        return

    summary_dir.mkdir(exist_ok=False)

    def getmax(df: pd.DataFrame):
        df = df.sort_values("step").copy()
        df["value"] = df["value"].cummax()
        return df[["tag", "fold", "step", "value"]]

    cdf: pd.DataFrame = (
        df[~df["tag"].str.startswith("Loss/")]
        .sort_values("step")
        .groupby(["tag", "fold"])
        .apply(getmax)
        .sort_values(["tag", "fold", "step"])
        .groupby(["tag", "step"])["value"]
        .mean()
        .reset_index()
    )

    with SummaryWriter(log_dir=summary_dir) as writer:
        for _, row in cdf.iterrows():
            writer.add_scalar(row.tag, row.value, row.step)

        max_values = cdf.groupby("tag")["value"].max()

        # hparams
        writer.add_hparams(
            {
                "lr": config.modelconfig.learning_rate,
                "weight_decay": config.modelconfig.weight_decay,
                "batch_size": config.Batch_SIZE,
                "model": config.MODEL_NAME,
                "loss_fn": config.modelconfig.loss_fn,
            },
            {f"~hparams/{tag}": value for tag, value in max_values.items()},
        )

        writer.flush()
