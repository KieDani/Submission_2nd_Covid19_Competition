import time
from typing import Callable, Sequence, Tuple, Union
import logging
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as _Dataset
import SimpleITK as sitk
import scipy.ndimage as ndimage
import json
import hashlib
import cv2

from training.data.elastic_deform import (
    elastic_deform_single_color,
    elastic_deform_single_color_3d,
)


def is_contiguous(arr: np.ndarray):
    return arr.flags["C_CONTIGUOUS"]


class BaseTransform:
    def __call__(self, item: dict):
        raise NotImplementedError("Custom transform must implement __call__")

    def _repr_params(self):
        raise NotImplementedError(
            f"_repr_params not implemented for {type(self).__name__}"
        )

    def __repr__(self):
        param_str = ", ".join(
            [f"{key}={value}" for key, value in self._repr_params().items()]
        )
        return f"{type(self).__name__}({param_str})"

    def convert_for_submission(self):
        return self


def to_config(config):
    if isinstance(config, BaseTransform):
        params = config._repr_params()
        params["_tfm"] = type(config).__name__
        return to_config(params)
    elif isinstance(config, Sequence) and not isinstance(config, str):
        return [to_config(item) for item in config]
    elif isinstance(config, dict):
        return {key: to_config(value) for key, value in config.items()}
    else:
        return config


def apply_transforms(transforms: Sequence[BaseTransform], item: dict):
    """Applies the given `transforms` iteratevily on `item`."""
    for tfm in transforms:
        item = tfm(item)
    return item


def profile_transforms(transforms: Sequence[BaseTransform], item: dict):
    measurements = []
    for tfm in transforms:
        t0 = time.time()
        item = tfm(item)
        t1 = time.time()
        measurements.append((tfm, t1 - t0))

    return measurements


class Dataset(_Dataset):
    def __init__(
        self,
        input: Sequence[dict],
        transforms: Sequence[BaseTransform],
        gpu_transforms=None,
    ):
        """
        @input: Sequence of dicts that is processed by the first transform.
        @transforms: Sequence of transforms that are applied one after the other. Must
        be a subclass of `BaseTransform`.
        @gpu_transforms: Sequence of callables that are applied on CUDA tensors.
        """
        self.input = input
        self.transforms = transforms
        self.gpu_transforms = [] if gpu_transforms is None else gpu_transforms

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx: int):
        return apply_transforms(self.transforms, self.input[idx])

    def tfm_config(self):
        return to_config(self.transforms)

    def gpu_transform(self, v_tensor: torch.Tensor):
        for tfm in self.gpu_transforms:
            v_tensor = tfm(v_tensor)
        return v_tensor


class Identity(BaseTransform):
    """Leaves the input data unchanged.
    Can be used for example to define optional transforms:
    ```
    tfm = OptionalTransform() if useit else Identity()
    ```
    """

    def __call__(self, item: dict):
        return item

    def _repr_params(self):
        return dict()


class LoadSimpleITKImage(BaseTransform):
    """Very similar to `LoadMHA` but expects an already loaded `SimpleITK.Image` on the
    image key. Replaces the SimpleITK image with a numpy array of the image voxel.
    Additional metadata (age and sex) is extracted as well.
    """

    def __init__(self, metadata=None, key="img"):
        if metadata is not None:
            logging.warn(
                "Metadata is now always enabled, you don't need to set it. (Keeping it for backwards compatibility for now)"
            )
        self.key = key

    def __call__(self, item: dict):
        d = dict(item)
        mha: sitk.Image = d[self.key]
        voxel = sitk.GetArrayFromImage(mha).T
        # do we need contiguousarray for performance?
        # it seems like the trade-off is too high though
        # voxel = np.ascontiguousarray(voxel)
        d[self.key] = voxel.astype(np.float32)

        if mha.HasMetaDataKey("PatientAge"):
            d["age"] = int(mha.GetMetaData("PatientAge").strip("Y"))
        else:
            d["age"] = 50
        if mha.HasMetaDataKey("PatientSex"):
            d["sex"] = mha.GetMetaData("PatientSex")
        else:
            d["sex"] = "O"
        return d

    def _repr_params(self):
        # for backwards compat, always set to True (so we don't need to rename the caches)
        return dict(metadata=True)


class LoadMha(LoadSimpleITKImage):
    """Loads an image file that SimpleITK can load and returns the whole voxel as a numpy
    array. Additional metadata (age and sex) is extracted as well.
    """

    def __init__(self, metadata=None, path_key="path", img_key="img"):
        super().__init__(key=img_key)
        self.path_key = path_key

    def __call__(self, item: dict):
        d = dict(item)
        d[self.key] = sitk.ReadImage(str(d[self.path_key]))
        return super().__call__(d)

    def _repr_params(self):
        # for backwards compat, always set to True (so we don't need to rename the caches)
        return dict(metadata=True)

    def convert_for_submission(self):
        # evalutils is used in the submission and already loads the images
        return LoadSimpleITKImage(key=self.key)


class LoadNpy(BaseTransform):
    def __init__(self, flip_dim1=False, path_key="path", img_key="img"):
        self.path_key = path_key
        self.img_key = img_key
        self.flip_dim1 = flip_dim1

    def __call__(self, item: dict):
        d = dict(item)
        d[self.img_key] = np.load(d[self.path_key]).astype(np.float32)
        if self.flip_dim1:
            d[self.img_key] = d[self.img_key][:, ::-1]
        return d

    def _repr_params(self):
        return dict(flip_dim1=self.flip_dim1)


class LoadJpegs(BaseTransform):
    def __init__(self, path_key="path", img_key="img", stack_axis=0):
        self.path_key = path_key
        self.img_key = img_key
        self.stack_axis = stack_axis

    def __call__(self, item: dict):
        d = dict(item)
        folder = Path(d[self.path_key])
        slices = []
        num_resolution = {}
        try:
            jpg_paths = [p for p in folder.glob("*.jpg") if p.stem.isdigit()]
            for path in sorted(jpg_paths, key=lambda p: int(p.stem)):
                img = cv2.imread(str(path))
                if img is None:
                    logging.info(f"Skip {path} (is None)")
                    continue
                assert img is not None, f"{path} is None"
                # the provided jpegs are RGB images which (should) contain the same values across channels
                assert (img[..., 0] == img[..., 1]).all() and (
                    img[..., 0] == img[..., 2]
                ).all()
                if img[..., 0].shape in num_resolution.keys():
                    num_resolution[img[..., 0].shape] += 1
                else:
                    num_resolution[img[..., 0].shape] = 1
                slices.append(img[..., 0])
            resolution = max(num_resolution)
            #remove slices with wrong resolution
            for idx, slice in enumerate(slices.copy()):
                if resolution != slice.shape: slices.pop(idx)

        except ValueError:
            raise ValueError(
                f"[LoadJpegs] Could not load JPEGs in {folder}. "
                "Maybe the filenames don't match? Check the error above for more information."
            )
        if len(slices) == 0:
            raise ValueError(f"LoadJpegs: Could not find any JPEGs in {folder}")
        d[self.img_key] = np.stack(slices).astype(np.float32)
        return d

    def _repr_params(self):
        return dict(stack_axis=self.stack_axis)


class AsType(BaseTransform):
    def __init__(self, dtype: str, key="img"):
        self.dtype = dtype
        self.key = key

    def __call__(self, item: dict):
        d = dict(item)
        d[self.key] = d[self.key].astype(self.dtype)
        return d

    def _repr_params(self):
        return dict(dtype=self.dtype)


class KeyFork(BaseTransform):
    "`true_transform` will always be triggered for the submission."

    def __init__(
        self, key: str, true_transform: BaseTransform, false_transform: BaseTransform
    ):
        self.key = key
        self.true_transform = true_transform
        self.false_transform = false_transform

    def __call__(self, item: dict):
        d = dict(item)
        if not d[self.key]:
            return self.false_transform(item)
        return self.true_transform(item)

    def _repr_params(self):
        return dict(
            key=self.key,
            true_transform=self.true_transform,
            false_transform=self.false_transform,
        )

    def convert_for_submission(self):
        return self.true_transform


class Select(BaseTransform):
    """Selects multiple values from the input dict and returns them as a tuple.
    This is normally the last transform of a dataset.
    """

    def __init__(self, *keys: Sequence[str], **kwargs):
        # kwargs is there to allow for multiple signature types:
        # 1) Select("key1", "key2")
        # 2) Select(keys=["key1", "key2"])
        if "keys" in kwargs:
            self.keys = kwargs["keys"]
        else:
            self.keys = keys

    def convert_for_submission(self):
        # for the submission only one image is processed at a time -> no need for tuples to generate batches
        # upside: no extra care for missing keys is required
        # (keys that are present during training but not for the submission)
        return Identity()

    def __call__(self, item):
        return tuple([item[k] for k in self.keys])

    def _repr_params(self):
        return dict(keys=self.keys)

    def __repr__(self):
        keys_str = ", ".join(self.keys)
        return f"{type(self).__name__}({keys_str})"


class Scale01(BaseTransform):
    """Rescales the intensity of the image to the interval [0, 1] and clips outliers if
    desired.
    """

    def __init__(self, lower: float, upper: float, clip=True, key="img"):
        """
        @lower: The value that should be mapped to 0
        @upper: The value that should be mapped to 1
        @clip: Wether to clip values that are outside of [lower, upper] to the respective
        edge value.
        """
        self.lower = lower
        self.upper = upper
        self.clip = clip
        self.key = key

    def __call__(self, item):
        d = dict(item)
        voxel = d[self.key]
        if self.clip:
            voxel = np.clip(voxel, self.lower, self.upper)
        voxel = (voxel - self.lower) / (self.upper - self.lower)
        d[self.key] = voxel
        return d

    def _repr_params(self):
        return dict(lower=self.lower, upper=self.upper, clip=self.clip)


class Clip(BaseTransform):
    def __init__(self, lower: float, upper: float, key="img"):
        self.lower = lower
        self.upper = upper
        self.key = key

    def __call__(self, item: dict):
        d = dict(item)
        d[self.key] = np.clip(d[self.key], self.lower, self.upper)
        return d

    def _repr_params(self):
        return dict(lower=self.lower, upper=self.upper)


class Zoom(BaseTransform):
    "Resizes the input image to the desired shape."

    def __init__(self, new_shape: Tuple[float, float, float], key="img"):
        self.new_shape = new_shape
        self.key = key

    def __call__(self, item):
        d = dict(item)
        voxel_tensor = d[self.key]
        scale_factors = (
            self.new_shape[0] / voxel_tensor.shape[0],
            self.new_shape[1] / voxel_tensor.shape[1],
            self.new_shape[2] / voxel_tensor.shape[2],
        )
        if is_contiguous(voxel_tensor):
            voxel_tensor = ndimage.zoom(voxel_tensor.T, scale_factors[::-1]).T
        else:
            voxel_tensor = ndimage.zoom(voxel_tensor, scale_factors)
        assert all(
            [self.new_shape[i] == voxel_tensor.shape[i] for i in range(3)]
        ), f"Shape mismatch! Expected {self.new_shape} but got {voxel_tensor.shape}"
        d[self.key] = voxel_tensor
        return d

    def _repr_params(self):
        return dict(new_shape=self.new_shape)


class DivideAge(BaseTransform):
    def __init__(self, max_age: int, key="age"):
        self.max_age = max_age
        self.key = key

    def __call__(self, item: dict):
        d = dict(item)
        if d[self.key] is None:
            return d
        d[self.key] = d[self.key] / self.max_age
        return d

    def _repr_params(self):
        return dict(max_age=self.max_age)


class Compose(BaseTransform):
    """Executes multiple transforms. Useful for grouping transforms."""

    def __init__(self, transforms: Sequence[Union[BaseTransform, nn.Module]]):
        self.transforms = transforms

    def __call__(self, item: dict):
        return apply_transforms(self.transforms, item)

    def _repr_params(self):
        return dict(transforms=self.transforms)


class Slice2D(BaseTransform):
    "Selects a random axial plane and its neighbour planes to return a RGB image-like array."

    def __init__(self, selection_start: float, selection_end: float, key="img"):
        self.key = key
        assert (
            selection_end >= selection_start
            and selection_start <= 1
            and selection_end <= 1
        )
        self.selection_start = selection_start
        self.selection_end = selection_end

    def __call__(self, item: dict):
        d = dict(item)
        img = d[self.key]
        pos = (
            random.random() * (self.selection_end - self.selection_start)
            + self.selection_start
        )
        axial_pos = int(pos * img.shape[2])
        d[self.key] = np.ascontiguousarray(img.T[axial_pos - 1 : axial_pos + 2])
        return d

    def _repr_params(self):
        return dict(
            selection_start=self.selection_start, selection_end=self.selection_end
        )


class EncodeSex(BaseTransform):
    "Takes the sex value (encoded as characters FMOA) and one-hot-encodes it."
    SEX_CATEGORIES = dict(F=0, M=1, O=2, A=2)

    def __init__(self, num_classes: int = 3, key="sex"):
        self.key = key
        self.num_classes = num_classes

    def __call__(self, item: dict):
        d = dict(item)
        if d[self.key] is None:
            return d

        sex = str(d[self.key])
        sex = self.SEX_CATEGORIES[sex]
        one_hot = F.one_hot(
            # torch's one_hot requires a tensor
            torch.tensor(sex),
            num_classes=self.num_classes,
        )
        d[self.key] = one_hot
        return d

    def _repr_params(self):
        return dict(num_classes=self.num_classes)


class Print(BaseTransform):
    """Can be used to debug datasets. `accessors` is a sequence of callables that can be
    used to extract arbitrary information from the current state. The return value will
    be printed. If an accessor is not a function, it is directly printed.

    For example, you can write the following:
    ```
    Print("Shape of img:", lambda d: d["img"].shape)
    ```
    """

    def __init__(self, *accessors: Sequence[Union[Callable, object]]):
        self.accessors = accessors

    def __call__(self, item):
        to_print = []
        for acc in self.accessors:
            if isinstance(acc, Callable):
                to_print.append(acc(item))
            else:
                to_print.append(acc)
        print(*to_print)
        return item

    def _repr_params(self):
        return dict()


class Rand(BaseTransform):
    "Applies `transform` with a probability of `prob`."

    def __init__(self, transform: BaseTransform, prob: float = 0.5):
        assert prob >= 0 and prob <= 1
        self.prob = prob
        self.transform = transform

    def __call__(self, item: dict):
        if random.random() < self.prob:
            return self.transform(item)
        return item

    def _repr_params(self):
        return dict(prob=self.prob, transform=self.transform)


class RandFork(BaseTransform):
    """Randomly applies either `main_transform` or `alternative_transform`
    (with probability `prob_alternative`).

    When used for submission, `main_transform` will *always* be used.
    """

    def __init__(
        self,
        main_transform: BaseTransform,
        alternative_transform: BaseTransform,
        prob_alternative: float = 0.5,
    ):

        self.main_transform = main_transform
        self.alternative_transform = alternative_transform
        self.prob_alternative = prob_alternative

    def __call__(self, item: dict):
        if random.random() < self.prob_alternative:
            return self.main_transform(item)
        else:
            return self.alternative_transform(item)

    def _repr_params(self):
        return dict(
            main_transform=self.main_transform,
            alternative_transform=self.alternative_transform,
            prob_alternative=self.prob_alternative,
        )

    def convert_for_submission(self):
        return self.main_transform


class GaussianFilter(BaseTransform):
    "Applies a Gaussian filter to an image to smooth it."

    def __init__(self, sigma_range: Tuple[float, float], key="img"):
        self.key = key
        self.sigma_range = sigma_range

    def __call__(self, item: dict):
        d = dict(item)
        voxel = d[self.key]
        sigma = np.random.uniform(self.sigma_range[0], self.sigma_range[1])
        # TODO: allow for more configuration?
        voxel = ndimage.gaussian_filter(voxel, sigma)
        d[self.key] = voxel
        return d

    def _repr_params(self):
        return dict(sigma_range=self.sigma_range)


class Crop(BaseTransform):
    def __init__(self, size: Tuple[float, float, float], key="img"):
        self.key = key
        self.size = size

    def __call__(self, item: dict):
        d = dict(item)
        voxel = d[self.key]
        v_shape = voxel.shape
        # second argument to randint is exclusive
        anchor = np.random.randint(
            (0, 0, 0),
            (
                v_shape[0] - self.size[0] + 1,
                v_shape[1] - self.size[1] + 1,
                v_shape[2] - self.size[2] + 1,
            ),
        )
        voxel = voxel[
            anchor[0] : anchor[0] + self.size[0],
            anchor[1] : anchor[1] + self.size[1],
            anchor[2] : anchor[2] + self.size[2],
        ]
        d[self.key] = voxel
        return d

    def _repr_params(self):
        return dict(size=self.size)


class Rotate(BaseTransform):
    def __init__(self, degrees_range: Tuple[float, float], key="img"):
        self.key = key
        self.degrees_range = degrees_range

    def __call__(self, item: dict):
        d = dict(item)
        voxel = d[self.key]
        degrees = np.random.uniform(self.degrees_range[0], self.degrees_range[1])
        voxel = ndimage.rotate(voxel, degrees, reshape=False)
        d[self.key] = voxel
        return d

    def _repr_params(self):
        return dict(degrees_range=self.degrees_range)


class RandomOrientation(BaseTransform):
    def __init__(self, key="img"):
        self.key = key

    def __call__(self, item: dict):
        d = dict(item)
        voxel = d[self.key]
        degree_steps = np.array([0, 90, 180, 270])

        # Randomly choose the orientation along each axis
        or1 = np.random.choice(degree_steps)
        or2 = np.random.choice(degree_steps)
        or3 = np.random.choice(degree_steps)

        voxel = ndimage.rotate(voxel, or1, reshape=False, axes=(0, 1))
        voxel = ndimage.rotate(voxel, or2, reshape=False, axes=(1, 2))
        voxel = ndimage.rotate(voxel, or3, reshape=False, axes=(0, 2))

        d[self.key] = voxel
        return d

    def _repr_params(self):
        return dict()



class AddNoise(BaseTransform):
    "Adds Gaussian noise"

    def __init__(
        self,
        mean_range: Tuple[float, float],
        std_range: Tuple[float, float],
        weight: float = 0.5,
        masked=True,
        key="img",
    ):
        """
        @weight: Weight of the noise in the resulting output tensor: `out = (1-w)*in + w*noise`
        @masked: If `True`, all areas where the input voxel is equal to the minimum over all voxels will not be modified.
        """
        self.key = key
        self.mean_range = mean_range
        self.std_range = std_range
        self.weight = weight
        self.masked = masked

    def __call__(self, item: dict):
        d = dict(item)
        voxels = d[self.key]
        mean = np.random.uniform(*self.mean_range)
        std = np.random.uniform(*self.std_range)

        # it doesn't matter if noise is transposed or not
        # but if voxels is transposed we'll get a nice speedup if noise is transposed too
        if is_contiguous(voxels):
            noise = np.random.normal(mean, std, voxels.shape).astype(voxels.dtype)
        else:
            noise = (
                np.random.normal(mean, std, voxels.shape[::-1]).astype(voxels.dtype).T
            )

        with_noise = ((1 - self.weight) * voxels) + (self.weight * noise)
        if self.masked:
            vmin = voxels.min()
            d[self.key] = np.where(voxels == vmin, vmin, with_noise)
        else:
            d[self.key] = with_noise
        return d

    def _repr_params(self):
        return dict(
            mean_range=self.mean_range,
            std_range=self.std_range,
            weight=self.weight,
            masked=self.masked,
        )


class Flip(BaseTransform):
    """Always flips the image on the given axis.

    You probably want to combine this transform with the `Rand`-Transform to add randomness.
    """

    def __init__(self, axis, key="img"):
        self.axis = axis
        self.key = key

    def __call__(self, item: dict):
        d = dict(item)
        img: np.ndarray = d[self.key]
        d[self.key] = np.flip(img, self.axis)
        return d

    def _repr_params(self):
        return dict(axis=self.axis)


class AxialFlip(Flip):
    "Use `Flip(axis=0)` instead!"

    def __init__(self, key="img"):
        super().__init__(0, key)


class ElasticDeform(BaseTransform):
    def __init__(
        self,
        sigma_range: Tuple[float, float],
        magnitude_range: Tuple[float, float],
        key="img",
    ):
        self.key = key
        self.sigma_range = sigma_range
        self.magnitude_range = magnitude_range

    def __call__(self, item: dict):
        d = dict(item)
        voxel = d[self.key]

        sigma = np.random.uniform(*self.sigma_range)
        magnitude = np.random.uniform(*self.magnitude_range)
        deformed = elastic_deform_single_color(voxel, alpha=magnitude, sigma=sigma)

        d[self.key] = deformed.astype(np.float32)
        return d

    def _repr_params(self):
        return dict(sigma_range=self.sigma_range, magnitude_range=self.magnitude_range)


class GPUElasticDeform:
    def __init__(
        self,
        sigma_range: Tuple[float, float],
        magnitude_range: Tuple[float, float],
    ):
        self.sigma_range = sigma_range
        self.magnitude_range = magnitude_range

    def __call__(self, batch: torch.Tensor):
        sigma = np.random.uniform(*self.sigma_range)
        magnitude = np.random.uniform(*self.magnitude_range)
        return elastic_deform_single_color_3d(
            batch, alpha=magnitude, sigma=sigma, device="cuda"
        )


class Unsqueeze(BaseTransform):
    def __init__(self, axis: int = 0, key="img"):
        self.key = key
        self.axis = axis

    def __call__(self, item):
        d = dict(item)
        voxel = d[self.key]
        voxel = np.expand_dims(voxel, axis=self.axis)
        d[self.key] = voxel
        return d

    def _repr_params(self):
        return dict(axis=self.axis)


class Normalize3D(BaseTransform):
    def __init__(self, mean, std, key="img"):
        self.mean = mean
        self.std = std
        self.key = key

    def __call__(self, item):
        d = dict(item)
        img: np.ndarray = d[self.key]
        img = (img - self.mean) / self.std
        d[self.key] = img
        return d

    def _repr_params(self):
        return dict(mean=self.mean, std=self.std)


class NumpyNegativeStrideCorrection(BaseTransform):
    def __init__(self, key="img"):
        self.key = key

    def __call__(self, item):
        d = dict(item)
        img: np.ndarray = d[self.key]
        img = np.flip(img, axis=0).copy()
        d[self.key] = img
        return d

    def _repr_params(self):
        return dict()


class Normalize2D(BaseTransform):
    def __init__(
        self,
        mean: Union[float, Tuple[float, float, float]] = (0.485, 0.456, 0.406),
        std: Union[float, Tuple[float, float, float]] = (0.229, 0.224, 0.225),
        key="img",
    ):
        """
        @mean: Defaults to ImageNet mean
        @std: Defaults to ImageNet std
        """
        assert isinstance(mean, Sequence) == isinstance(std, Sequence)

        if isinstance(mean, Sequence):
            assert len(mean) == 3 and len(std) == 3
            self.mean = tuple(mean)
            self.std = tuple(std)
        else:
            self.mean = mean
            self.std = std

        self.key = key

    def __call__(self, item):
        d = dict(item)
        x = d[self.key]

        if isinstance(self.mean, tuple):
            # if a tuple is set, we must aggregate

            # for grayscale, pytorch uses:
            # l_img = (0.2989 * r + 0.587 * g + 0.114 * b).to(img.dtype)
            # https://github.com/pytorch/vision/blob/59c723cb45d0f8ab897cc7836d408e9fdde4b552/torchvision/transforms/functional_tensor.py#L152
            w0, w1, w2 = 0.2989, 0.587, 0.114
            m0, m1, m2 = self.mean
            s0, s1, s2 = self.std

            # since we only have grayscale images, use the same image for all three channels (and weight using grayscale weights)
            # z = (w0 * (x - m0) / s0) + (w1 * (x - m1) / s1) + (w2 * (x - m2) / s2)
            # and rewritten to be more efficient:
            z = x * (w0 / s0 + w1 / s1 + w2 / s2) - (
                w0 * m0 / s0 + w1 * m1 / s1 + w2 * m2 / s2
            )
            d[self.key] = z
        else:
            d[self.key] = (x - self.mean) / self.std

        return d

    def _repr_params(self):
        return dict(mean=self.mean, std=self.std)


class FileCache(BaseTransform):
    """Checks if the child transforms have been cached to a file and loads from that file
    if possible. .npz is used for the file format.
    """

    def __init__(
        self,
        transforms: Sequence[BaseTransform],
        cache_root: Union[str, Path],
        keys: Sequence[str],
        no_save: bool = False,
    ):
        """
        Creates a new `FileCache` transform. A file named "cache_transforms.json" will be
        created inside the cache directory that contains the configuration of the child
        transforms.

        The cache directory will be determined based on the
        transforms to cache. The transform settings are encoded as a JSON string and
        hashed. This hash is the name of the cache directory and uniquely identifies this
        exact transform configuration used for the file cache.

        @key: Is used to determine a unique identifier to locate the respective files in
        the cache directory. Can either be a string or a function that maps a dict to a
        string.
        """

        self.transforms = transforms
        self.keys = keys

        self.cache_root = Path(cache_root)
        # if using auto_dir, cache_root will be modified but for _repr_params we need the original setting
        self._orig_cache_root = self.cache_root

        # sort_keys=True is important to return the same hash every time
        config_str = json.dumps(self._hashable_config(), default=str, sort_keys=True)
        hash = hashlib.sha256(config_str.encode()).hexdigest()
        self.cache_root = self.cache_root / hash

        logging.info(f"Using cache directory {self.cache_root}")

        # no side-effects should occur during __init__
        # directory and config dump are created on the first save of an item
        self._first_save = True
        self.no_save = no_save

    def _save(self, item: dict, cache_path: Path):
        if self.no_save:
            return
        # no side-effects should occur during __init__
        # directory and config dump are created on the first save of an item
        if self._first_save:
            self._first_save = False
            logging.warn(
                f"Could not find cached version for {cache_path.stem}! Building it now"
            )
            self.cache_root.mkdir(exist_ok=True, parents=True)

            if not (self.cache_root / "cache_transforms.json").exists():
                with open(
                    self.cache_root / "cache_transforms.json", "w"
                ) as config_file:
                    json.dump(
                        self._hashable_config(), config_file, indent=2, default=str
                    )

        # make sure that Path objects don't land in the cache file (don't work with allow_pickle=False)
        cleaned_item = {}
        for k, v in item.items():
            if isinstance(v, Path):
                cleaned_item[k] = str(v)
            else:
                cleaned_item[k] = v
        np.savez(cache_path, **cleaned_item)

    def _load(self, cache_path: Path):
        with np.load(cache_path) as npz:
            return dict(npz)

    def __call__(self, item: dict):
        d = dict(item)

        cache_key = "-".join(str(d[k]) for k in self.keys)
        assert "/" not in cache_key, "key must not result in parts containing /"
        cache_path = self.cache_root / f"{cache_key}.npz"

        if cache_path.exists():
            cached = self._load(cache_path)
            for k, v in cached.items():
                d[k] = v
        else:
            d = apply_transforms(self.transforms, d)

            self._save(d, cache_path)

        return d

    def _hashable_config(self):
        # cache_root must be removed from the config or the hash would depend on the data path
        # which may change depending on the user
        params = self._repr_params()
        params["_tfm"] = "FileCache"
        del params["cache_root"]
        return to_config(params)

    def _repr_params(self):
        return dict(
            transforms=self.transforms,
            cache_root=self._orig_cache_root,
            keys=self.keys,
        )

    def convert_for_submission(self):
        # no caching required during submission
        return Compose(transforms=self.transforms)


# TFM_CLS_LIST is a list of all available transforms (they are exported in __all__ and can be loaded from a config file)
TFM_CLS_LIST = [
    AddNoise,
    AsType,
    AxialFlip,
    Clip,
    Compose,
    Crop,
    Dataset,
    DivideAge,
    ElasticDeform,
    EncodeSex,
    FileCache,
    Flip,
    GaussianFilter,
    Identity,
    KeyFork,
    LoadJpegs,
    LoadMha,
    LoadNpy,
    LoadSimpleITKImage,
    Normalize3D,
    Normalize2D,
    NumpyNegativeStrideCorrection,
    Print,
    Rand,
    RandFork,
    Rotate,
    Scale01,
    Select,
    Slice2D,
    Unsqueeze,
    Zoom,
]
__all__ = [c.__name__ for c in TFM_CLS_LIST]


def tfms_from_config(config: Union[dict, Sequence], submission_mode=False):
    """Loads data transforms from a config

    @submission_mode: If activated, each transform's `convert_to_submission()` method is
    called which replaces some transforms to be ready for submission input.
    """
    name_to_tfm = {c.__name__: c for c in TFM_CLS_LIST}

    if isinstance(config, dict):
        TYPE_KEY = "_tfm"
        if TYPE_KEY in config:
            name = config[TYPE_KEY]
            params = {
                key: tfms_from_config(value, submission_mode)
                for key, value in config.items()
                if key != TYPE_KEY
            }
            tfm_cls = name_to_tfm[name]
            try:
                tfm = tfm_cls(**params)
                if submission_mode:
                    return tfm.convert_for_submission()
                else:
                    return tfm
            except:
                raise RuntimeError(
                    f"Could not create transform {name} with parameters: {params}. See the stacktrace above for more information."
                )
        else:
            output = {}
            for key, value in config.items():
                output[key] = tfms_from_config(value, submission_mode)
            return output
    elif isinstance(config, Sequence) and not isinstance(config, str):
        return [tfms_from_config(row, submission_mode) for row in config]
    else:
        return config
