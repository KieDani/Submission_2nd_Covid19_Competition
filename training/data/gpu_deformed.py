from pathlib import Path
from training.data.data_api import *
from training.data.data_api import GPUElasticDeform, Normalize3D, NumpyNegativeStrideCorrection
from training.data.load_labels_from_split import load_labels_from_split


def get_gpudeformed_dataset(dataconfig):
    data_root = Path(dataconfig.data_path)

    labels = load_labels_from_split(data_root, dataconfig.patients)

    img_size = (dataconfig.img_size, dataconfig.img_size, dataconfig.img_size)
    middle_size = (
        dataconfig.cache_img_size,
        dataconfig.cache_img_size,
        dataconfig.cache_img_size,
    )

    loader_orig_res = FileCache(
        [
            LoadMha(),
            Zoom(img_size),
            Scale01(-1000, 500),
        ],
        cache_root=dataconfig.precompute_path,
        # note that variant is missing here - it does not matter when looking at original images
        keys=("patient",),
        auto_dir=True,
    )
    loader_orig_crop = FileCache(
        [
            LoadMha(),
            Zoom(middle_size),
            Scale01(-1000, 500),
        ],
        cache_root=dataconfig.precompute_path,
        keys=("patient",),
        auto_dir=True,
    )

    if dataconfig.is_validation:
        # validation transforms
        loader = loader_orig_res
        trainonly_tfms = []
        select_keys = ["img", "age", "sex", "inf", "sev", "patient"]
    else:
        # training transforms
        loader = RandFork(
            main_transform=loader_orig_res,
            alternative_transform=Compose([loader_orig_crop, Crop(img_size)]),
        )
        trainonly_tfms = [
            Compose([Rand(Flip(0), 0.25), Rand(Flip(1), 0.25), Rand(Flip(2), 0.25)])
            if dataconfig.flip
            else Identity(),
            Rand(
                Rotate((-dataconfig.rotate, dataconfig.rotate)),
                prob=dataconfig.rotate_prob,
            ),
            Rand(
                GaussianFilter(
                    sigma_range=dataconfig.blur_sigma,
                ),
                prob=dataconfig.blur_prob,
            ),
            Rand(
                AddNoise(
                    mean_range=dataconfig.noise_mean,
                    std_range=dataconfig.noise_std,
                    weight=dataconfig.noise_weight,
                ),
                prob=dataconfig.noise_prob,
            ),
        ]
        select_keys = ["img", "age", "sex", "inf", "sev"]

    if dataconfig.deform_prob > 0 and not dataconfig.is_validation:
        # normally Rand() would be applied to a dict but it does not rely on the type
        # so we can abuse it here for randomness
        gpu_transforms = [
            Rand(
                GPUElasticDeform(
                    sigma_range=dataconfig.deform_sigma,
                    magnitude_range=dataconfig.deform_alpha,
                ),
                prob=dataconfig.deform_prob,
            )
        ]
    else:
        gpu_transforms = []

    return Dataset(
        input=labels[["patient", "path", "inf", "sev"]].to_dict(orient="records"),
        transforms=[
            # loader is without elastic deformation
            loader,
            Clip(0.0, 1.0),
            Normalize3D(mean=dataconfig.mean, std=dataconfig.std) if dataconfig.do_normalization else Identity(),
            # transforms that are only applied for the training dataset:
            *trainonly_tfms,
            NumpyNegativeStrideCorrection(),
            Unsqueeze(),
            EncodeSex(),
            DivideAge(max_age=85),
            Select(*select_keys),
        ],
        gpu_transforms=gpu_transforms,
    )
