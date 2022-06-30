import torch
from pathlib import Path
import torchvision.transforms as tv_tfm
import numpy as np

from training.data.data_api import *
from training.data.data_api import BaseTransform
from training.data.load_labels_from_split import load_labels_from_split

class TensorTransform(BaseTransform):
    def __init__(self, transform, key="img"):
        self.transform = transform
        self.key = key

    def __call__(self, item: dict):
        img = torch.tensor(item[self.key])
        d = dict(item)
        img = self.transform(img)
        d[self.key] = np.array(img)
        return d

    def _repr_params(self):
        return dict(transform=type(self.transform).__name__)
        

def get_slice2d_dataset(dataconfig):
    data_root = Path(dataconfig.data_path)
    labels = load_labels_from_split(data_root, dataconfig.patients)


    return Dataset(
        input=labels[["patient", "path", "inf", "sev"]].to_dict(orient="records"),
        transforms=[
            FileCache(
                [
                    LoadMha(),
                    Scale01(-1100, 300),
                    Slice2D(0.45, 0.6),
                    Zoom((3, 224, 224)),
                ],
                auto_dir=True,
                cache_root=data_root/"data"/"auto_cache",
                keys=["patient"],
            ),
            # don't use the custom Normalize here!
            TensorTransform(tv_tfm.Compose([
                tv_tfm.RandomHorizontalFlip(),
                tv_tfm.RandomVerticalFlip(),
                # tv_tfm.RandomRotation((-30, 30)),
                # tv_tfm.RandomPerspective(0.2),
                tv_tfm.ColorJitter(hue=0.3, saturation=0.2, brightness=0.1, contrast=0.2),
                # tv_tfm.RandomAdjustSharpness(2),
                tv_tfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])),
            EncodeSex(num_classes=2),
            DivideAge(max_age=85),
            Select("img", "age", "sex", "inf", "sev"),
        ],
    )
