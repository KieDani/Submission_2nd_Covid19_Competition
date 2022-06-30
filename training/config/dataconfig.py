import logging
from pathlib import Path
from typing import Sequence, Union
import pandas as pd

from training.data.data2d import get_slice2d_dataset
from training.data.deformed import get_deformed_dataset
from training.data.gpu_deformed import get_gpudeformed_dataset
from training.data.mia import get_mia_dataset
from training.data.mosmed import get_mosmed_dataset
from training.data.noaug import get_noaug_dataset
from training.data.hust import get_hust_dataset


class Slice2DDataConfig:
    def __init__(self, config, patients: Sequence[int] = None, is_validation=False):
        self.data_path = config.DATA_PATH
        self.patients = patients
        self.is_validation = is_validation


class DeformedDataConfig:
    def __init__(
        self,
        config,
        patients: Sequence[int] = None,
        is_validation=False,
        cache_path=None,
    ):
        self.data_path = config.DATA_PATH
        self.patients = patients
        self.is_validation = is_validation
        if cache_path is None:
            self.precompute_path = cache_path
        else:
            self.precompute_path = Path(self.data_path) / "data" / "auto_cache"
        logging.info(f"Cache path: {self.precompute_path}")

        self.cache_img_size = 256
        self.img_size = 224

        self.num_repeats = 1  # 4

        # configs for augmentation
        # AxialFlip
        self.flip = True

        # Rotate
        self.rotate_prob = 0.2
        self.rotate = 30

        # GaussianFilter
        self.blur_prob = 0.2
        self.blur_sigma = (0.6, 0.8)

        # AddNoise
        self.noise_prob = 0.2
        self.noise_mean = (0.5, 0.5)
        self.noise_std = (0.3, 1)
        self.noise_weight = 0.03

        self.deform = True


class GPUDeformedDataConfig:
    def __init__(
        self,
        config,
        patients: Sequence[int] = None,
        is_validation=False,
        cache_path=None,
    ):
        self.data_path = config.DATA_PATH
        self.patients = patients
        self.is_validation = is_validation
        if cache_path is not None:
            self.precompute_path = cache_path
        else:
            self.precompute_path = Path(self.data_path) / "data" / "auto_cache"
        logging.info(f"Cache path: {self.precompute_path}")

        self.cache_img_size = 256
        self.img_size = 224

        # configs for augmentation
        # AxialFlip
        self.flip = True

        # Rotate
        self.rotate_prob = 0.2
        self.rotate = 30

        # GaussianFilter
        self.blur_prob = 0.2
        self.blur_sigma = (0.6, 0.8)

        # AddNoise
        self.noise_prob = 0.2
        self.noise_mean = (0.5, 0.5)
        self.noise_std = (0.3, 1)
        self.noise_weight = 0.03

        self.deform_prob = 0.5
        self.deform_sigma = (35, 35)
        self.deform_alpha = (1, 7)

        # self.do_normalization = False
        self.do_normalization = True
        self.mean = 0.3110
        self.std = 0.3154


class MiaDataConfig:
    def __init__(
        self,
        config,
        patients: Sequence[int] = None,
        is_validation=False,
        cache_path=None,
    ):
        self.data_path = config.DATA_PATH
        self.patients = patients
        self.is_validation = is_validation
        if cache_path is not None:
            self.precompute_path = cache_path
        else:
            self.precompute_path = Path(self.data_path) / "data" / "auto_cache"
        logging.info(f"Cache path: {self.precompute_path}")

        self.cache_img_size = 256
        self.img_size = 224

        # configs for augmentation
        # AxialFlip
        self.flip = True

        # Rotate
        self.rotate_prob = 0.2
        self.rotate = 30

        # GaussianFilter
        self.blur_prob = 0.2
        self.blur_sigma = (0.6, 0.8)

        # AddNoise
        self.noise_prob = 0.2
        self.noise_mean = (0.5, 0.5)
        self.noise_std = (0.3, 1)
        self.noise_weight = 0.03

        self.deform_prob = 0.5
        self.deform_sigma = (35, 35)
        self.deform_alpha = (1, 7)

        # self.do_normalization = False
        self.do_normalization = True
        self.mean = 0.3971
        self.std = 0.3207

        # Give a random orientation to the body
        self.orientation_prob = 0.0 # Would have suggested 0.25 as a probability


class MosmedConfig:
    def __init__(self, config, patients=None, is_validation=False):
        self.data_path = config.DATA_PATH
        self.patients = patients
        self.is_validation = is_validation
        self.img_size = 224

        # configs for augmentation
        # AxialFlip
        self.flip = True

        # Rotate
        self.rotate_prob = 0.2
        self.rotate = 30

        # GaussianFilter
        self.blur_prob = 0.2
        self.blur_sigma = (0.6, 0.8)

        # AddNoise
        self.noise_prob = 0.2
        self.noise_mean = (0.5, 0.5)
        self.noise_std = (0.3, 1)
        self.noise_weight = 0.03

        self.deform_prob = 0.5
        self.deform_sigma = (35, 35)
        self.deform_alpha = (1, 7)

        self.mosmed_path = Path(config.DATA_PATH) / "data" / "mosmed"
        self.mosmed_inf_level = 1
        self.mosmed_sev_level = 3

        self.output_study = False

        self.do_normalization = False
        self.mean = 0.2926
        self.std = 0.3119


class HustConfig:
    def __init__(self, config, patients=None, is_validation=False):
        self.data_path = config.DATA_PATH
        self.patients = patients
        self.is_validation = is_validation
        self.img_size = 224

        # configs for augmentation
        # AxialFlip
        self.flip = True

        # Rotate
        self.rotate_prob = 0.2
        self.rotate = 30

        # GaussianFilter
        self.blur_prob = 0.2
        self.blur_sigma = (0.6, 0.8)

        # AddNoise
        self.noise_prob = 0.2
        self.noise_mean = (0.5, 0.5)
        self.noise_std = (0.3, 1)
        self.noise_weight = 0.03

        self.deform_prob = 0.5
        self.deform_sigma = (35, 35)
        self.deform_alpha = (1, 7)

        # self.hust_path = Path(config.DATA_PATH) / "data" / "hust"
        # TODO move hust in the correct directory...
        self.hust_path = Path(config.DATA_PATH) / "hust"

        self.output_study = False

        self.do_normalization = False
        self.mean = 0.7454
        self.std = 0.2002


class NoAugConfig:
    def __init__(self, config, patients=None, is_validation=False, cache_path=None):
        self.data_path = config.DATA_PATH
        self.patients = patients
        self.is_validation = is_validation
        if cache_path is not None:
            self.precompute_path = cache_path
        else:
            self.precompute_path = Path(self.data_path) / "data" / "auto_cache"
        logging.info(f"Cache path: {self.precompute_path}")

        self.img_size = 224


def get_miaval_dataconfig(config, cache_path):
    if config.split in ['cv5_infonly.csv', 'eccv.csv']:
        official = pd.read_csv(config.get_split_path("eccv.csv"))
    else:
        official = pd.read_csv(config.get_split_path("official.csv"))
    val = official[official["Set"] == "val"]
    # skip = [
    #     "train_cov19d/covid/ct_scan_31",
    #     "train_cov19d/covid/ct_scan_47",
    #     "train_cov19d/non-covid/ct_scan899",
    #     "train_cov19d/non-covid/ct_scan988",
    #     "train_cov19d/non-covid/ct_scan_292",
    #     "train_cov19d/non-covid/ct_scan_354",
    #     "train_cov19d/non-covid/ct_scan_537",
    #     "train_cov19d/non-covid/ct_scan_853",
    #     "validation_cov19d/covid/ct_scan_101",
    #     "validation_cov19d/covid/ct_scan_18",
    #     "validation_cov19d/covid/ct_scan_40",
    #     "validation_cov19d/covid/ct_scan_46",
    #     "validation_cov19d/covid/ct_scan_48",
    #     "validation_cov19d/non-covid/ct_scan221",
    #     "validation_cov19d/non-covid/ct_scan250",
    #     "validation_cov19d/non-covid/ct_scan_15",
    # ]
    skip = []
    val = val[~val["Path"].isin(skip)]
    return MiaDataConfig(
        config,
        patients=val["patient"].tolist(),
        is_validation=True,
        cache_path=cache_path,
    )


def get_dataconfig(
    config, patients: Sequence[int] = None, is_validation=False, cache_path=None
):
    "Provide `patients` to only select a subset of all cases. Useful for train/val/test split."

    if patients is not None and len(patients) == 0:
        # patients is empty which means we should skip the dataset
        return None

    dataset_name = config.DATA_NAME
    if dataset_name == "mia":
        return MiaDataConfig(
            config,
            patients,
            is_validation=is_validation,
            cache_path=cache_path,
        )
    if dataset_name == "deformed":
        return DeformedDataConfig(
            config,
            patients=patients,
            is_validation=is_validation,
            cache_path=cache_path,
        )
    if dataset_name == "gpudeformed":
        return GPUDeformedDataConfig(
            config,
            patients=patients,
            is_validation=is_validation,
            cache_path=cache_path,
        )
    if dataset_name == "slice2d":
        return Slice2DDataConfig(config, patients=patients, is_validation=is_validation)
    if dataset_name == "noaug":
        return NoAugConfig(
            config,
            patients=patients,
            is_validation=is_validation,
            cache_path=cache_path,
        )
    if dataset_name == "mosmed_stoic" or dataset_name == "mosmed":
        return MosmedConfig(config, patients=patients, is_validation=is_validation)
    if dataset_name == "hust_stoic" or dataset_name == "hust":
        return HustConfig(config, patients=patients, is_validation=is_validation)
    raise KeyError(f"No dataset found for {dataset_name}")


def get_cvsplit_patients(index_path: Union[pd.DataFrame, str, Path], fold_id: int):
    if isinstance(index_path, pd.DataFrame):
        index = index_path
    else:
        index = pd.read_csv(index_path)

    val = index[index["cv"] == fold_id]
    train = index.drop(index=val.index)

    return dict(
        train=train["patient"].tolist(),
        val=val["patient"].tolist(),
    )


def get_dataset(dataconfig):
    if dataconfig is None:
        # patients is empty which means we should skip the dataset
        return []

    if isinstance(dataconfig, DeformedDataConfig):
        return get_deformed_dataset(dataconfig)
    if isinstance(dataconfig, GPUDeformedDataConfig):
        return get_gpudeformed_dataset(dataconfig)
    if isinstance(dataconfig, Slice2DDataConfig):
        return get_slice2d_dataset(dataconfig)
    if isinstance(dataconfig, NoAugConfig):
        return get_noaug_dataset(dataconfig)
    if isinstance(dataconfig, MosmedConfig):
        return get_mosmed_dataset(dataconfig)
    if isinstance(dataconfig, HustConfig):
        return get_hust_dataset(dataconfig)
    if isinstance(dataconfig, MiaDataConfig):
        return get_mia_dataset(dataconfig)

    raise KeyError(f"No dataset found for {dataconfig}")
