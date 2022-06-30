from pathlib import Path
import pandas as pd
from training.data.data_api import *
from training.data.load_labels_from_split import load_labels_from_split


def get_deformed_dataset(dataconfig):
    data_root = Path(dataconfig.data_path)

    labels = load_labels_from_split(data_root, dataconfig.patients)

    frames = []
    for r in range(dataconfig.num_repeats):
        df = labels.copy()
        df["variant"] = r
        frames.append(df)

    labels = pd.concat(frames, ignore_index=True)

    img_size = (dataconfig.img_size, dataconfig.img_size, dataconfig.img_size)
    middle_size = (dataconfig.cache_img_size, dataconfig.cache_img_size, dataconfig.cache_img_size)

    loader_orig_res = FileCache(
        [
            LoadMha(),
            Zoom(img_size),
            Scale01(-1000, 500),
        ],
        cache_root=dataconfig.precompute_path,
        # note that variant is missing here - it does not matter when looking at original images
        keys=("patient", ),
        auto_dir=True,
    )
    loader_orig_crop = FileCache(
        [
            LoadMha(),
            Zoom(middle_size),
            Scale01(-1000, 500),
        ],
        cache_root=dataconfig.precompute_path,
        keys=("patient", ),
        auto_dir=True,
    )

    elastic_deformation = ElasticDeform(sigma_range=(10, 40), magnitude_range=(30, 60))
    loader_deform_res = FileCache(
        [
            LoadMha(),
            Zoom(img_size),
            Scale01(-1000, 500),
            elastic_deformation,
        ],
        cache_root=dataconfig.precompute_path,
        keys=("patient", "variant"),
        auto_dir=True,
    )
    loader_deform_crop = FileCache(
        [
            LoadMha(),
            Zoom(middle_size),
            Scale01(-1000, 500),
            elastic_deformation,
        ],
        cache_root=dataconfig.precompute_path,
        keys=("patient", "variant"),
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
            # original image
            main_transform=RandFork(
                main_transform=loader_orig_res,
                alternative_transform=Compose([loader_orig_crop, Crop(img_size)]),
            ),
            # elastic deformation
            alternative_transform=RandFork(
                main_transform=loader_deform_res,
                alternative_transform=Compose([loader_deform_crop, Crop(img_size)]),
            ),
            prob_alternative=0.5 if dataconfig.deform else 0,
        )
        trainonly_tfms = [
            Compose([Rand(Flip(0), 0.25), Rand(Flip(1), 0.25), Rand(Flip(2), 0.25)]) if dataconfig.flip else Identity(),
            Rand(Rotate((-dataconfig.rotate, dataconfig.rotate)), prob=dataconfig.rotate_prob),
            Rand(GaussianFilter(
                sigma_range=dataconfig.blur_sigma,
            ), prob=dataconfig.blur_prob),
            Rand(AddNoise(
                mean_range=dataconfig.noise_mean,
                std_range=dataconfig.noise_std,
                weight=dataconfig.noise_weight,
            ), prob=dataconfig.noise_prob),
        ]
        select_keys = ["img", "age", "sex", "inf", "sev"]

    return Dataset(
        input=labels[["patient", "path", "inf", "sev", "variant"]].to_dict(orient="records"),
        transforms=[
            # loader is either with or without elastic deformation
            loader,
            # transforms that are only applied for the training dataset:
            *trainonly_tfms,
            Clip(0.0, 1.0),
            Unsqueeze(),
            EncodeSex(),
            DivideAge(max_age=85),
            Select(*select_keys),
        ],
    )
