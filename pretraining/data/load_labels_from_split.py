from typing import Sequence, Union
from pathlib import Path
import pandas as pd

def load_labels_from_split(data_root: Path, patients: Union[Sequence[int], None]):
    data_root = Path(data_root)
    labels: pd.DataFrame = pd.read_csv(data_root/"metadata"/"reference.csv")
    labels.rename(columns={
        "probCOVID": "inf",
        "probSevere": "sev",
        "PatientID": "patient",
    }, inplace=True)

    if patients is not None:
        patients = pd.Series(patients, name="patient")
        labels = pd.merge(patients, labels)

    labels["path"] = labels["patient"].apply(lambda pid: str(data_root/"data"/"mha"/f"{pid}.mha"))

    return labels
