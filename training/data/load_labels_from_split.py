from typing import Sequence, Union
from pathlib import Path
import pandas as pd

def load_labels_from_split(data_root: Path, patients: Union[Sequence[int], None]):
    data_root = Path(data_root)
    labels: pd.DataFrame = pd.read_csv(data_root/"my_reference.csv")
    labels.rename(columns={
        "Inf": "inf",
        "Sev": "sev",
        "Path": "path",
    }, inplace=True)
    labels["inf"] -= 1
    labels["sev"] -= 1

    # this code was hotfixed to work with the ECCV challenge
    # it does not work with STOIC anymore :(
    labels["patient"] = labels["path"].str.replace("/", "-")
    labels["path"] = [str(data_root / p) for p in labels["path"]]

    if patients is not None:
        labels = labels[labels["patient"].isin(patients)]
        assert len(labels) == len(patients)

    return labels
