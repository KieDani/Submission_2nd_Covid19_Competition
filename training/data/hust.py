import math
import pandas as pd
from training.data.data_api import *
from training.data.data_api import GPUElasticDeform, NumpyNegativeStrideCorrection
import torch.nn.functional as F
import torch


def get_hust_dataset(dataconfig):
    # Hust
    hust_entries = []
    df = pd.read_csv(dataconfig.hust_path  / 'hust_metadata_cleaned.csv')
    for npy_path in dataconfig.hust_path.glob("*.npy"):
        patient_number = int(npy_path.name.strip('.npy'))
        assert patient_number == int(df['Patient'][patient_number - 1])
        age, sex, sev_level = df['Age'][patient_number - 1], df['Gender'][patient_number - 1], df['Outcome'][patient_number - 1]
        sex = 0 if sex == 'Female' else 1
        one_hot_sex = F.one_hot(torch.tensor(sex), num_classes=3)

        hust_entries.append(
            {
                "patient": f"hust-{npy_path.stem.split('_')[-1]}",
                "age": age,
                "sex": one_hot_sex,
                "inf": 1 if sev_level in ['Regular', 'Mild', 'Severe', 'Critically ill'] else 0,
                "sev": 1 if sev_level in ['Severe', 'Critically ill'] else 0,
                "study": sev_level,
                "path": npy_path,
            }
        )
    assert len(hust_entries) > 0

    labels = pd.DataFrame(hust_entries)

    img_size = (dataconfig.img_size, dataconfig.img_size, dataconfig.img_size)

    select_keys = ["img", "age", "sex", "inf", "sev"]
    if dataconfig.output_study:
        select_keys.append("study")
    if dataconfig.is_validation:
        # validation transforms
        trainonly_tfms = []
        select_keys.append("patient")
    else:
        # training transforms
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
        input=labels[["patient", "study", "age", "sex", "path", "inf", "sev"]].to_dict(
            orient="records"
        ),
        transforms=[
            LoadNpy(flip_dim1=True),
            Zoom(img_size),
            Scale01(-1000, 500),
            Clip(0.0, 1.0),
            Normalize3D(mean=dataconfig.mean, std=dataconfig.std) if dataconfig.do_normalization else Identity(),
            # transforms that are only applied for the training dataset:
            *trainonly_tfms,
            NumpyNegativeStrideCorrection(),
            Unsqueeze(),
            DivideAge(max_age=85),
            Select(*select_keys),
        ],
        gpu_transforms=gpu_transforms,
    )
