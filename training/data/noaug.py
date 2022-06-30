from pathlib import Path
from training.data.data_api import *
from training.data.load_labels_from_split import load_labels_from_split


def get_noaug_dataset(dataconfig):
    data_root = Path(dataconfig.data_path)

    labels = load_labels_from_split(data_root, dataconfig.patients)

    if dataconfig.is_validation:
        select_keys = ["img", "age", "sex", "inf", "sev", "patient"]
    else:
        select_keys = ["img", "age", "sex", "inf", "sev"]

    return Dataset(
        input=labels[["patient", "path", "inf", "sev"]].to_dict(orient="records"),
        transforms=[
            FileCache(
                [
                    LoadMha(),
                    Zoom(
                        (dataconfig.img_size, dataconfig.img_size, dataconfig.img_size)
                    ),
                    Scale01(-1000, 500),
                ],
                cache_root=dataconfig.precompute_path,
                keys=("patient",),
                auto_dir=True,
            ),
            Clip(0.0, 1.0),
            Unsqueeze(),
            EncodeSex(),
            DivideAge(max_age=85),
            Select(*select_keys),
        ],
    )
