from datetime import datetime
import random
import argparse
import json
import os
from pathlib import Path
import time
from typing import Sequence
import torch
import torch.optim
import torch.utils.data as data
from tqdm import tqdm, trange
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
import time
import logging
from training.data.mosmed import get_mosmed_dataset
from training.data.zip_loader import ZipLoader

import training.evaluate
from training.data.siamese import SiameseLoader
from training.misc_utilities.tensorboard import SummaryWriter, log_cv_tensorboard_summary, log_cv_tensorboard_bestsummary
from training.misc_utilities import determinism
from training.misc_utilities.plot import plot_conf_mat
from training.config.config import BaseConfig
from training.config.modelconfig import get_model, get_optimizer
from training.config.dataconfig import get_dataset
from training.time_iterator import itertime
from training.misc_utilities.git import write_git_log
from training.loss import BinaryCrossEntropySevOnly, assert_siamese_compatibility, get_loss
from training.misc_utilities import deactivate_batchnorm


def plot_batch(imgs: torch.Tensor, writer: SummaryWriter):
    fig, axs = plt.subplots(nrows=1, ncols=len(imgs), figsize=(len(imgs) * 2, 2))
    for ax, tensor in zip(axs.flatten(), imgs):
        ax.imshow(tensor[0, tensor.size(1) // 2])
    writer.add_figure("first_batch", fig)


def parse_args():
    parser = argparse.ArgumentParser(description='Train STOIC network')
    # general

    parser.add_argument('--gpu',
                        default="1",
                        help='gpu id',
                        type=str)

    parser.add_argument('--model',
                        help='Model used for training, possible are: convnext, hyperconvnext, resnet, slice2d, uniformer',
                        type=str,
                        default="convnext")

    parser.add_argument('--cv',
                        help='Use cross-validation',
                        dest='cv',
                        action='store_true')
    parser.add_argument('--no-cv',
                        dest='cv',
                        action='store_false')
    parser.set_defaults(cv=False)
    parser.add_argument('--reset-seed',
                        action='store_true',
                        default=False,
                        help='Call set_deterministic for every fold')

    parser.add_argument('--nick',
                        default=None,
                        help='Prepend a nickname to the output directory',
                        type=str)

    parser.add_argument('--num_steps',
                        default=1,
                        help='steps for gradient accumulation',
                        type=int)

    parser.add_argument('--num_trained_stages',
                        default=4,
                        help='number of stages with requires_grad=True',
                        type=int)

    parser.add_argument('--no-tqdm',
                        default=False,
                        action='store_true')

    parser.add_argument("--miatest",
                        default=False,
                        action="store_true")


    args = parser.parse_args()

    return args


def nan2none(tensor: torch.Tensor):
    nans = tensor.isnan()
    if nans.any():
        assert nans.all(), "Only some entries are NaN. Conversion to None failed"
        return None
    return tensor

def metric_a_better_than_b(a, b):
    # if best_metrics is None or metrics["auc_sev2_ema"] > best_metrics["auc_sev2_ema"]:
    return b is None or a["f1_macro_sev_ema"] > b["f1_macro_sev_ema"] + 0.0001 or a["f1_macro_inf_ema"] > b["f1_macro_inf_ema"] + 0.0001

class Trainer:
    def __init__(self, config: BaseConfig, log_directory=None, datasets=None, dataconfigs=None, no_tqdm=False):
        """
        @log_directory: If set, the trainer puts all logs inside this directory instead of generating one.
        @datasets: If set, use the specified datasets instead of deriving them from the
        `config`. Must be set iff `dataconfigs` is set.
        @dataconfigs: Overwrites `config.dataconfigs`. Must be set iff `datasets` is set.
        """
        assert (datasets is None) == (dataconfigs is None), "Either set both datasets and dataconfigs or set none of them"

        self.config = config
        self.no_tqdm = no_tqdm
        print("------ using model: {} ------".format(config.MODEL_NAME))

        # overwrite existing dataconfigs if desired
        # (usually only required for Ray Tune)
        if datasets is None:
            self.datasets = {phase: get_dataset(cfg) for phase, cfg in self.config.dataconfigs.items()}
        else:
            # if using Ray Tune, datasets can be shared across processes which enables us to load the entire dataset into RAM
            self.config.dataconfigs = dataconfigs
            self.datasets = datasets

        # setup a custom output directory if not specified
        if log_directory is None:
            self.output_dir = config.get_output_dir()
            self.output_dir.mkdir(exist_ok=True, parents=True)
        else:
            # if using Ray Tune the output directory can't be changed, therefore we need
            # the ability to set the output directory explicitly
            self.output_dir = Path(log_directory)
        logging.info(f"Output directory is: {self.output_dir}")

        # dump config as JSON
        with open(self.output_dir / "config.json", "w") as config_file:
            json.dump(self.config.to_dict(), config_file, indent=2, default=str)

        # write data transforms config
        with open(self.output_dir/"dataset.json", "w") as data_json:
            json.dump(
                {phase: dataset.tfm_config() for (phase, dataset) in self.datasets.items()},
                data_json, indent=2, default=str,
            )

        # write data inputs
        for phase, dataset in self.datasets.items():
            if len(dataset) > 0:
                pd.DataFrame(dataset.input).to_csv(self.output_dir/f"input_{phase}.csv", index=False)

        modelcfg = self.config.modelconfig
        self.model = get_model(self.config)
        if self.config.use_ema:
            self.alpha = 0.995 if self.config.modelconfig.num_classes == 2 else 0.99
            print('Alpha value for EMA:', self.alpha)
            self.model_ema = get_model(self.config)
            with torch.no_grad():
                for name, param in self.model_ema.named_parameters():
                    param.data = self.model.state_dict()[name]
        # self.model_uses_metadata = getattr(modelcfg, "use_metadata", False)
        # for ECCV challenge: no metadata is available
        self.model_uses_metadata = False

        if torch.device(self.config.DEVICE).type == "cpu":
            logging.warning("Computation will run on the CPU.\nIs this intended?")

        write_git_log(self.output_dir/"git.log")
        self.writer = SummaryWriter(self.output_dir)
        # Tensorboard uses markdown for text -> wrap the JSON inside a code fence
        self.writer.add_text("config", "``` json\n" + json.dumps(self.config.to_dict(), indent=2, default=str) + "\n```")

        if getattr(modelcfg, "ce3_weighted", False):
            counts = [0, 0, 0]
            for inp in self.datasets["train"].input:
                counts[int(inp["inf"] + inp["sev"])] += 1
            counts = torch.tensor(counts)
            ce3_weight = (counts.float().mean() / counts).to(self.config.DEVICE)
            logging.info(f"Cross Entropy Weights: {ce3_weight}")
        else:
            ce3_weight = None
        self.loss_fn = get_loss(modelcfg.loss_fn, modelcfg.pos_weight, ce3_weight=ce3_weight)
        # make sure the loss is compatible with siamese mode
        assert_siamese_compatibility(self.loss_fn, self.config.modelconfig.siamese)

        self.optimizer = get_optimizer(modelcfg, self.model)

        if modelcfg.lr_decay is not None and (modelcfg.cosine_decay is None):
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=modelcfg.lr_decay["every_num_epochs"],
                gamma=modelcfg.lr_decay["gamma"]
            )
        elif modelcfg.cosine_decay is not None:
            # Compute total number of training steps
            max_epochs = self.config.MAX_EPOCHS if self.config.MAX_EPOCHS_LRSCHEDULER is None else self.config.MAX_EPOCHS_LRSCHEDULER
            T_max = self.datasets["train"].__len__() // self.config.Batch_SIZE * max_epochs

            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=T_max,
                eta_min=modelcfg.cosine_decay["lr_min"]
            )

        if modelcfg.siamese:
            loader_cls = SiameseLoader
        else:
            loader_cls = data.DataLoader

        self.train_loader = loader_cls(
            dataset=self.datasets["train"], num_workers=self.config.WORKERS, batch_size=self.config.Batch_SIZE, shuffle=True,
            worker_init_fn=determinism.seed_worker
        )

        self.val_loader = data.DataLoader(
            dataset=self.datasets["val"], num_workers=self.config.WORKERS, batch_size=self.config.Batch_SIZE,
            worker_init_fn=determinism.seed_worker
        )
        if "test" in self.datasets:
            self.test_loader = data.DataLoader(
                dataset=self.datasets["test"], num_workers=self.config.WORKERS, batch_size=self.config.Batch_SIZE,
                worker_init_fn=determinism.seed_worker
            )
        else:
            self.test_loader = None

        if self.config.mosmed_config is not None:
            assert not modelcfg.siamese, "mosmed does not work with siamese"
            mosmed_loader = data.DataLoader(
                dataset=get_mosmed_dataset(self.config.mosmed_config),
                num_workers=self.config.WORKERS,
                batch_size=self.config.Batch_SIZE,
                shuffle=True,
                worker_init_fn=determinism.seed_worker,
            )
            self.train_loader = ZipLoader(self.train_loader, mosmed_loader)
            logging.info("ZipLoader for MosMed enabled")

        self.model = self.model.to(self.config.DEVICE)
        if self.config.use_ema: self.model_ema = self.model_ema.to(self.config.DEVICE)

    def update_ema(self):
        alpha = self.alpha
        with torch.no_grad():
            for name, param in self.model_ema.named_parameters():
                param.data = alpha * param.data + (1 - alpha) * self.model.state_dict()[name].data

    def epoch(self, epoch_num: int):
        "Executes one single epoch (including evaluation step)"
        # measure time to detect IO vs GPU bottleneck
        min_load_time = float("inf")
        min_train_time = float("inf")

        self.model.train()
        deactivate_batchnorm.set_all_bn_eval(self.model)

        max_sev_ratio_epoch = getattr(self.config.modelconfig, "loss_max_sev_ratio_epoch", None)
        if max_sev_ratio_epoch is not None:
            self.loss_fn.set_moving_ratio(epoch_num, max_sev_ratio_epoch)

        ce3_moving_epoch = getattr(self.config.modelconfig, "ce3_moving_epoch", None)
        if ce3_moving_epoch is not None:
            self.loss_fn.set_moving_weights(epoch_num, ce3_moving_epoch)

        num_at_once = self.config.Batch_SIZE // self.config.NUM_STEPS

        for batch_idx, (load_time, sample) in enumerate(tqdm(itertime(self.train_loader), total=len(self.train_loader), leave=True, position=0, desc="Training one Epoch", disable=self.no_tqdm)):
            if self.config.modelconfig.siamese:
                (v_tensor, age, sex, inf_gt, sev_gt), (neg_v_tensor, neg_age, neg_sex, neg_inf_gt, neg_sev_gt) = sample
            else:
                v_tensor, inf_gt, sev_gt = sample

            if epoch_num == 0 and batch_idx == 0:
                plot_batch(v_tensor, self.writer)

            # https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#use-parameter-grad-none-instead-of-model-zero-grad-or-optimizer-zero-grad
            self.optimizer.zero_grad(set_to_none=True)

            train_start_time = time.time()
            loss_sum = 0
            running_loss_sev0 = 0
            running_loss_sev1 = 0
            sev0_counter = 0
            sev1_counter = 0

            max_i = min(v_tensor.shape[0], neg_v_tensor.shape[0]) if self.config.modelconfig.siamese else v_tensor.shape[0]
            loss_scaling = max((max_i // num_at_once), 1)
            for i in range(0, max_i, num_at_once):
                step = min(i + num_at_once, max_i)

                v_ten, s_gt, i_gt = v_tensor[i:step], sev_gt[i:step], inf_gt[i:step]
                v_ten, s_gt, i_gt = v_ten.to(self.config.DEVICE), s_gt.to(self.config.DEVICE), i_gt.to(self.config.DEVICE)
                if self.model_uses_metadata:
                    a, s = age[i:step], sex[i:step]
                    a, s = a.to(self.config.DEVICE), s.to(self.config.DEVICE)
                    # mosmed produces batches without metadata (will be set to NaN)
                    # metadata must be set to None to disable in the forward pass
                    a = nan2none(a)
                    s = nan2none(s)
                else:
                    a = None
                    s = None

                v_ten = self.train_loader.dataset.gpu_transform(v_ten)
                output = self.model(v_ten, a, s, train_stages=self.config.modelconfig.train_stages)

                if self.config.modelconfig.siamese:
                    # select subset for gradient accumulation
                    v_ten = neg_v_tensor[i:step].to(self.config.DEVICE)
                    s_gt = neg_sev_gt[i:step].to(self.config.DEVICE)
                    i_gt = neg_inf_gt[i:step].to(self.config.DEVICE)
                    # metadata
                    if self.model_uses_metadata:
                        a = neg_age[i:step].to(self.config.DEVICE)
                        s = neg_sex[i:step].to(self.config.DEVICE)
                    else:
                        a = None
                        s = None

                    v_ten = self.train_loader.dataset.gpu_transform(v_ten)
                    # forward negative data
                    output_neg = self.model(v_ten, a, s, train_stages=self.config.modelconfig.train_stages)
                    # stack for siam loss
                    output = torch.stack([output[:, 1], output_neg[:, 1]], dim=1)

                # when using siamese, i_gt and s_gt don't matter but are passed anyway for compatibility
                l = self.loss_fn(output, i_gt, s_gt) / loss_scaling
                loss_sum += l.item()
                l.backward()

                with torch.no_grad():
                    l0, l1 = self.loss_fn.partial_loss(output, i_gt, s_gt)
                    running_loss_sev0 += l0.cpu().item() / loss_scaling
                    running_loss_sev1 += l1.cpu().item() / loss_scaling
                sev1_c = s_gt.sum().cpu().item()
                sev0_counter += len(s_gt) - sev1_c
                sev1_counter += sev1_c

            self.optimizer.step()

            if epoch_num < self.config.modelconfig.decay_lr_until:
                # Use cosine scheduler for each training step
                if (self.config.modelconfig.lr_decay is None) and (self.config.modelconfig.cosine_decay is not None):
                    self.scheduler.step()

            if self.config.use_ema:
                self.update_ema()

            train_end_time = time.time()

            iter_num = epoch_num * len(self.train_loader) + batch_idx
            self.writer.add_scalar("Loss/train", loss_sum, iter_num)
            if sev0_counter > 0:
                self.writer.add_scalar("Loss/train_sev=0", running_loss_sev0, iter_num)
            if sev1_counter > 0:
                self.writer.add_scalar("Loss/train_sev=1", running_loss_sev1, iter_num)

            min_load_time = min(load_time, min_load_time)
            min_train_time = min(min_train_time, train_end_time - train_start_time)

        if self.config.modelconfig.siamese:
            val_loss_fn = BinaryCrossEntropySevOnly()
        else:
            val_loss_fn = self.loss_fn
        metrics = evaluate.evaluate(self.val_loader, self.model, val_loss_fn, self.model_uses_metadata, device=self.config.DEVICE, no_tqdm=self.no_tqdm, num_at_once=self.config.NUM_STEPS)
        self.dump_predictions(metrics["all_data"], "pred", epoch_num)
        if self.config.use_ema:
            ema_metrics = evaluate.evaluate(self.val_loader, self.model_ema, val_loss_fn, self.model_uses_metadata, device=self.config.DEVICE, no_tqdm=self.no_tqdm, num_at_once=self.config.NUM_STEPS)
            self.dump_predictions(ema_metrics["all_data"], "emapred", epoch_num)
            ema_metrics = {
                "acc_sev_ema": ema_metrics["acc_sev"],
                "f1_macro_sev_ema": ema_metrics["f1_macro_sev"],
                "acc_inf_ema": ema_metrics["acc_inf"],
                "f1_macro_inf_ema": ema_metrics["f1_macro_inf"],
                "auc_inf_ema": ema_metrics["auc_inf"],
                "auc_sev_ema": ema_metrics["auc_sev"],
                "auc_sev2_ema": ema_metrics["auc_sev2"],
                "val_loss_ema": ema_metrics["loss"],
                "val_loss_ema_sev0": ema_metrics["loss_sev0"],
                "val_loss_ema_sev1": ema_metrics["loss_sev1"],
            }
        else:
            ema_metrics = {}

        all_data = metrics["all_data"]
        inf_gt = all_data["inf_gt"]
        sev_gt = all_data["sev_gt"]
        inf_pred = all_data["inf_pred"]
        sev_pred = all_data["sev_pred"]

        self.writer.add_histogram("hist_sev2=0/val", sev_pred[(inf_gt == 1) & (sev_gt == 0)], global_step=epoch_num + 1)
        self.writer.add_histogram("hist_sev2=1/val", sev_pred[(inf_gt == 1) & (sev_gt == 1)], global_step=epoch_num + 1)

        self.writer.add_pr_curve("pr_inf/val", inf_gt, inf_pred, global_step=epoch_num + 1)
        self.writer.add_pr_curve("pr_sev/val", sev_gt, sev_pred, global_step=epoch_num + 1)
        self.writer.add_pr_curve("pr_sev2/val", sev_gt[inf_gt == 1], sev_pred[inf_gt == 1], global_step=epoch_num + 1)

        self.writer.add_figure("conf_mat/val", plot_conf_mat(sev_gt[inf_gt == 1], sev_pred[inf_gt == 1]), global_step=epoch_num + 1)
        self.writer.add_figure("conf_mat_inf/val", plot_conf_mat(inf_gt, inf_pred), global_step=epoch_num + 1)
        # plot ROC curve
        # self.writer.add_figure("ROC/sev", RocCurveDisplay.from_predictions(sev_gt, sev_pred).figure_, global_step=epoch_num + 1)
        # self.writer.add_figure("ROC/sev2", RocCurveDisplay.from_predictions(sev_gt[inf_gt == 1], sev_pred[inf_gt == 1]).figure_, global_step=epoch_num + 1)

        if epoch_num < self.config.modelconfig.decay_lr_until:
            # Carry out a lr-scheduler step
            if (self.config.modelconfig.lr_decay is not None) and (self.config.modelconfig.cosine_decay is None):
                self.scheduler.step()

        if self.test_loader is not None:
            test_metrics = evaluate.evaluate(self.test_loader, self.model, val_loss_fn, self.model_uses_metadata, device=self.config.DEVICE, no_tqdm=self.no_tqdm, num_at_once=self.config.NUM_STEPS)
            if self.config.use_ema:
                ema_test_metrics = evaluate.evaluate(self.test_loader, self.model_ema, val_loss_fn, self.model_uses_metadata, device=self.config.DEVICE, no_tqdm=self.no_tqdm, num_at_once=self.config.NUM_STEPS)

            inf_gt = test_metrics["all_data"]["inf_gt"]
            inf_pred = test_metrics["all_data"]["inf_pred"]
            sev_gt = test_metrics["all_data"]["sev_gt"]
            sev_pred = test_metrics["all_data"]["sev_pred"]

            # prepare dicts to be integrated into the final big metric dict
            test_metrics = { "test_" + k: v for (k, v) in test_metrics.items()}
            ema_test_metrics = { "ema_test_" + k: v for (k, v) in ema_test_metrics.items()}

            self.writer.add_figure("conf_mat/test", plot_conf_mat(sev_gt[inf_gt == 1], sev_pred[inf_gt == 1]), global_step=epoch_num + 1)
            self.writer.add_figure("conf_mat_inf/test", plot_conf_mat(inf_gt, inf_pred), global_step=epoch_num + 1)
        else:
            # Test set is not actually user. To avoid dereferencing a non-existent
            # variable we construct a fake test_metrics dict

            test_metrics = {
                "auc_inf": -1.0,
                "auc_sev": -1.0,
                "auc_sev2": -1.0,
                "loss": -1.0,
                "loss_sev0": -1.0,
                "loss_sev1": -1.0,
                "all_data": {
                    "inf_gt": -1.0,
                    "inf_pred": -1.0,
                    "sev_gt": -1.0,
                    "sev_pred": -1.0,
                    "patients": -1.0,
                }, }

            ema_test_metrics = test_metrics.copy()

            test_metrics = {"test_" + k: v for (k, v) in test_metrics.items()}
            ema_test_metrics = {"ema_test_" + k: v for (k, v) in ema_test_metrics.items()}


        del all_data, inf_gt, sev_gt, inf_pred, sev_pred

        # metrics must be returned as a dictionary for Ray Tune
        # to show them in tensorboard without Ray Tune, the metrics are written in the .run() method
        return {
            "acc_inf": metrics["acc_inf"],
            "f1_macro_inf": metrics["f1_macro_inf"],
            "acc_sev": metrics["acc_sev"],
            "f1_macro_sev": metrics["f1_macro_sev"],
            "auc_inf": metrics["auc_inf"],
            "auc_sev": metrics["auc_sev"],
            "auc_sev2": metrics["auc_sev2"],
            "val_loss": metrics["loss"],
            "val_loss_sev0": metrics["loss_sev0"],
            "val_loss_sev1": metrics["loss_sev1"],

            "min_load_time": min_load_time,
            "min_train_time": min_train_time,
            **ema_metrics,

            **test_metrics,
            **ema_test_metrics,
        }

    def save_checkpoint(self, checkpoint_dir, epoch_num=None, metrics=None):
        """Saves the current weights using `torch.save()` in a file named `{checkpoint_dir}/checkpoint.pt`.

        If `val_loss` is given, store the loss in the checkpoint and also update the best weights stored in the checkpoint.
        """
        # if using Ray Tune, checkpoint_dir is a temporary directory

        checkpoint = {
            "model": self.model.state_dict(),
            "model_ema": self.model_ema.state_dict() if self.config.use_ema else None,
            "optimizer": self.optimizer.state_dict(),
            "metrics": metrics,
            # TODO: scheduler,
            # config is already dumped to JSON but we also put it inside the checkpoint for easy reuse of the checkpoint
            "config": self.config.to_dict(),
            # only save validation transforms because only they are needed for inference again
            "data_transforms": self.datasets["val"].tfm_config(),
        }

        torch.save(checkpoint, Path(checkpoint_dir) / f"checkpoint-{epoch_num}.pt")

        # delete old checkpoints
        # this will probably not work with ray tune
        assert self.config.keep_additional_checkpoints != 0, "Keeping zero checkpoints is currently not supported"
        if self.config.keep_additional_checkpoints > 0:
            best_metric = None
            best_epoch = None
            pathlist = {}
            for cpath in Path(checkpoint_dir).glob("checkpoint-*.pt"):
                epoch = int(cpath.stem.split("-")[-1])
                pathlist[epoch] = cpath
                ckp = torch.load(cpath, map_location="cpu")
                if metric_a_better_than_b(a=ckp["metrics"], b=best_metric):
                    best_metric = ckp["metrics"]
                    best_epoch = epoch
            del ckp
            keep_right = (self.config.keep_additional_checkpoints - 1) // 2
            keep_left = self.config.keep_additional_checkpoints - 1 - keep_right
            for epoch, path in pathlist.items():
                if epoch != epoch_num and (epoch < best_epoch - keep_left or epoch > best_epoch + keep_right):
                    path.unlink()


    def load_checkpoint(self, checkpoint_dir, load_best=False, load_epoch=None):
        """Set `load_best` to True to load the best weights (by validation loss). If set
        to False, will load the last weights.
        """
        # if using Ray Tune, checkpoint_dir is a temporary directory
        checkpoint_dir = Path(checkpoint_dir)
        if load_epoch is None:
            check_paths = sorted(checkpoint_dir.glob("checkpoint-*.pt"), key=lambda p: int(p.stem.split("-")[1]))
            checkpoint = torch.load(check_paths[-1])
            if load_best:
                raise NotImplementedError("Looks up auc_sev2_ema at the moment which does not work for MIA")
                for path in check_paths:
                    # load to CPU for speedup
                    tmp = torch.load(path, map_location=torch.device("cpu"))
                    # TODO: make it work without ema?
                    if tmp["metrics"]["auc_sev2_ema"] > checkpoint["metrics"]["auc_sev2_ema"]:
                        checkpoint = tmp
        else:
            checkpoint = torch.load(checkpoint_dir/f"checkpoint-{load_epoch}.pt")

        self.load_state(checkpoint)
        # TODO: scheduler

    def load_state(self, checkpoint):
        self.model.load_state_dict(checkpoint["model"])
        self.model.to(self.config.DEVICE)
        if self.config.use_ema:
            self.model_ema.load_state_dict(checkpoint["model_ema"])
            self.model_ema.to(self.config.DEVICE)
        self.optimizer.load_state_dict(checkpoint["optimizer"])

    def run(self):
        """Runs all epochs as specified in the config.
        This method is not called when using Ray Tune for hyper parameter search.
        """
        best_metrics = None
        best_epoch = 0
        use_babysitter = self.config.babysitter_decay is not None and self.config.babysitter_grace_steps is not None
        grace_reference_epoch = 0
        for epoch_num in tqdm(range(self.config.MAX_EPOCHS), disable=self.no_tqdm):
            metrics = self.epoch(epoch_num)

            self.save_checkpoint(self.output_dir, epoch_num=epoch_num, metrics=metrics)
            # write metrics for the current validation results
            self.write_tensorboard_metrics(metrics, epoch_num)

            if metric_a_better_than_b(a=metrics, b=best_metrics):
                best_metrics = metrics
                best_epoch = epoch_num
                # this model was the best so far, remember for babysitting
                grace_reference_epoch = epoch_num
            else:
                # early stopping?
                if self.config.early_stopping > 0 and epoch_num >= self.config.early_stopping_grace_epochs and epoch_num - best_epoch > self.config.early_stopping:
                    logging.info("Early stopping")
                    break

                if use_babysitter and epoch_num > grace_reference_epoch + self.config.babysitter_grace_steps:
                    grace_reference_epoch = epoch_num
                    # load previous model state and update learning rate and optimizer
                    self.load_checkpoint(self.output_dir, load_epoch=grace_reference_epoch)
                    self.config.modelconfig.learning_rate /= self.config.babysitter_decay
                    self.optimizer = get_optimizer(self.config.modelconfig, self.model)

        # hparams: to compare final statistics for this training run
        self.write_tensorboard_hparams(best_metrics)

    def continue_run_until(self, stop_epoch: int):
        "Load previous state and run training until epoch `stop_epoch` has completed."
        use_babysitter = self.config.babysitter_decay is not None and self.config.babysitter_grace_steps is not None
        assert not use_babysitter, "Babysitter is not supported in run_continue"

        saved_epochs = [int(p.stem.split("-")[-1]) for p in self.output_dir.glob("checkpoint-*.pt")]
        if len(saved_epochs) == 0:
            last_saved_epoch = -1
        else:
            last_saved_epoch = max(saved_epochs)
            # load previous state
            prev_checkpoint = torch.load(self.output_dir / f"checkpoint-{last_saved_epoch}.pt")
            self.load_state(prev_checkpoint)

        for epoch_num in trange(last_saved_epoch + 1, stop_epoch + 1, disable=self.no_tqdm, unit="epoch"):
            metrics = self.epoch(epoch_num)
            self.save_checkpoint(self.output_dir, epoch_num=epoch_num, metrics=metrics)
            self.write_tensorboard_metrics(metrics, epoch_num)


    def write_tensorboard_metrics(self, metrics, epoch_num):
        metric_map = [
            ("Accuracy_Inf/val", "acc_inf"),
            ("F1_Macro_Inf/val", "f1_macro_inf"),
            ("Accuracy_Sev/val", "acc_sev"),
            ("F1_Macro_Sev/val", "f1_macro_sev"),
            ("ROC_AUC/val_inf", "auc_inf"),
            ("ROC_AUC/val_sev", "auc_sev"),
            ("ROC_AUC/val_sev2", "auc_sev2"),
            ("Loss/val", "val_loss"),
            ("Loss/val_sev=0", "val_loss_sev0"),
            ("Loss/val_sev=1", "val_loss_sev1"),
            # test
            ("Accuracy_Inf/test", "test_acc_inf"),
            ("F1_Macro_Inf/test", "test_f1_macro_inf"),
            ("Accuracy_Sev/test", "test_acc_sev"),
            ("F1_Macro_Sev/test", "test_f1_macro_sev"),
            ("ROC_AUC/test_inf", "test_auc_inf"),
            ("ROC_AUC/test_sev", "test_auc_sev"),
            ("ROC_AUC/test_sev2", "test_auc_sev2"),
            ("Loss/test", "test_loss"),
            ("Loss/test_sev=0", "test_loss_sev0"),
            ("Loss/test_sev=1", "test_loss_sev1"),
        ]
        for tbname, metric_name in metric_map:
            if metric_name in metrics:
                self.writer.add_scalar(tbname, metrics[metric_name], epoch_num + 1)

        ema_metric_map = [
            ("Accuracy_Inf/val_ema", "acc_inf_ema"),
            ("F1_Macro_Inf/val_ema", "f1_macro_inf_ema"),
            ("Accuracy_Sev/val_ema", "acc_sev_ema"),
            ("F1_Macro_Sev/val_ema", "f1_macro_sev_ema"),
            ("ROC_AUC/val_inf_ema", "auc_inf_ema"),
            ("ROC_AUC/val_sev_ema", "auc_sev_ema"),
            ("ROC_AUC/val_sev2_ema", "auc_sev2_ema"),
            ("Loss/val_ema", "val_loss_ema"),
            ("Loss/val_ema_sev=0", "val_loss_ema_sev0"),
            ("Loss/val_ema_sev=1", "val_loss_ema_sev1"),
            # test
            ("Accuracy_Inf/test_ema", "ema_test_acc_inf"),
            ("F1_Macro_Inf/test_ema", "ema_test_f1_macro_inf"),
            ("Accuracy_Sev/test_ema", "ema_test_acc_sev"),
            ("F1_Macro_Sev/test_ema", "ema_test_f1_macro_sev"),
            ("ROC_AUC/test_inf_ema", "ema_test_auc_inf"),
            ("ROC_AUC/test_sev_ema", "ema_test_auc_sev"),
            ("ROC_AUC/test_sev2_ema", "ema_test_auc_sev2"),
            ("Loss/test_ema", "ema_test_loss"),
            ("Loss/test_ema_sev=0", "ema_test_loss_sev0"),
            ("Loss/test_ema_sev=1", "ema_test_loss_sev1"),
        ]

        if self.config.use_ema:
            for tbname, metric_name in ema_metric_map:
                if metric_name in metrics:
                    self.writer.add_scalar(tbname, metrics[metric_name], epoch_num + 1)

        self.writer.add_scalar("Time/data", metrics["min_load_time"], epoch_num + 1)
        self.writer.add_scalar("Time/gpu", metrics["min_train_time"], epoch_num + 1)

        if self.config.babysitter_decay is not None and self.config.babysitter_grace_steps is not None:
            self.writer.add_scalar("X/lr", self.config.modelconfig.learning_rate, epoch_num + 1)

        if self.config.cv_enabled:
            log_cv_tensorboard_summary(cv_root=self.output_dir.parent, num_folds=self.config.num_folds)
            log_cv_tensorboard_bestsummary(self.output_dir.parent, self.config.num_folds, self.config)

    def write_tensorboard_hparams(self, best_metrics):
        if self.config.use_ema:
            ema_metrics = {
                "~hparams/acc_inf_ema": best_metrics["acc_inf"],
                "~hparams/f1_macro_inf_ema": best_metrics["f1_macro_inf_ema"],
                "~hparams/acc_sev_ema": best_metrics["acc_sev"],
                "~hparams/f1_macro_sev_ema": best_metrics["f1_macro_sev_ema"],
                "~hparams/auc_inf_ema": best_metrics["auc_inf_ema"],
                "~hparams/auc_sev_ema": best_metrics["auc_sev_ema"],
                "~hparams/auc_sev2_ema": best_metrics["auc_sev2_ema"],
                "~hparams/val_loss_ema": best_metrics["val_loss_ema"],
            }
        else:
            ema_metrics = {}

        self.writer.add_hparams({
            "lr": self.config.modelconfig.learning_rate,
            "weight_decay": self.config.modelconfig.weight_decay,
            "batch_size": self.config.Batch_SIZE,
            "model": self.config.MODEL_NAME,
            "loss_fn": self.config.modelconfig.loss_fn,
        }, {
            "~hparams/acc_inf": best_metrics["acc_inf"],
            "~hparams/f1_macro_inf": best_metrics["f1_macro_inf"],
            "~hparams/acc_sev": best_metrics["acc_sev"],
            "~hparams/f1_macro_sev": best_metrics["f1_macro_sev"],
            "~hparams/auc_inf": best_metrics["auc_inf"],
            "~hparams/auc_sev": best_metrics["auc_sev"],
            "~hparams/auc_sev2": best_metrics["auc_sev2"],
            "~hparams/val_loss": best_metrics["val_loss"],
            **ema_metrics,
        })

    def dump_predictions(self, all_data, prefix: str, epoch: int):
        df = pd.DataFrame(all_data)
        df.to_csv(self.output_dir / f"{prefix}-{epoch}.csv", index=False)


def main():
    determinism.set_deterministic()
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    logging.getLogger().setLevel(logging.INFO)

    config = BaseConfig(
        args.model,
        num_steps=args.num_steps,
        cv_enabled=args.cv,
        num_trained_stages=args.num_trained_stages,
        nickname=args.nick,
        split="cv5_infonly.csv",
        evaluate_official_val=args.miatest,
    )
    run(config, no_tqdm=args.no_tqdm, reset_deterministic=args.reset_seed)

    logging.info("Done")


def fold_has_completed(config: BaseConfig):
    # the last checkpoint is always stored
    # if the last checkpoint is >= the number epochs in the config, training is done
    epochs = [int(p.stem.split("-")[1]) for p in config.get_output_dir().glob("*.pt")]
    if len(epochs) == 0:
        return False
    last_epoch = max(epochs)
    return last_epoch >= config.MAX_EPOCHS - 1


def run_cv_alternating(config: BaseConfig, no_tqdm=False):
    config.get_output_dir().mkdir(parents=True)
    # generate different seeds for each epoch
    random.seed(1055)
    seeds = []
    for fold in range(config.num_folds):
        fold_seeds = []
        for epoch in range(config.MAX_EPOCHS):
            fold_seeds.append(random.randrange(1 << 31))
        seeds.append(fold_seeds)
    with open(config.get_output_dir() / "seeds.json", "w") as seeds_file:
        json.dump(seeds, seeds_file)

    assert config.cv_enabled, "run_cv_alternating only works with cross-validation"
    for epoch in trange(config.MAX_EPOCHS, unit="epoch"):
        for fold in trange(config.num_folds, unit="fold"):
            determinism.set_deterministic(seed=seeds[fold][epoch])
            config.set_fold(fold)
            trainer = Trainer(config, no_tqdm=no_tqdm)
            trainer.continue_run_until(epoch)


def run(config: BaseConfig, no_tqdm=False, reset_deterministic=False, reverse_cv=False):
    for fold in trange(config.num_folds, unit="fold"):
        if reverse_cv:
            fold = config.num_folds - fold - 1
        if reset_deterministic:
            determinism.set_deterministic()
        config.set_fold(fold)
        if fold_has_completed(config):
            logging.warn(f"Fold {fold} appears to be complete, skipping")
            if not reset_deterministic:
                logging.warn("Fold will be skipped but determinism is not reset!")
            continue
        fold_trainer = Trainer(config, no_tqdm=no_tqdm)
        fold_trainer.run()


def multirun(configs: Sequence[BaseConfig], no_tqdm=False, reset_deterministic=False, reverse_cv=False):
    """Run multiple configs but always complete one fold for all configs then the next
    fold and so on.
    """
    max_folds = max([c.num_folds for c in configs])
    for fold in trange(max_folds, unit="fold"):
        if reverse_cv:
            fold = max_folds - fold - 1
        for cfg in tqdm(configs, unit="config", desc=f"Fold {fold}"):
            if fold >= cfg.num_folds:
                continue
            if reset_deterministic:
                determinism.set_deterministic()
            cfg.set_fold(fold)
            if fold_has_completed(cfg):
                logging.warn(f"Fold {fold} appears to be complete, skipping")
                if not reset_deterministic:
                    logging.warn("Fold will be skipped but determinism is not reset!")
                continue
            fold_trainer = Trainer(cfg, no_tqdm=no_tqdm)
            fold_trainer.run()


if __name__ == "__main__":
    main()
