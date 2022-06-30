# script to build cache for a given dataconfig
# this should run faster than simply iterating over the dataset because unnecessary transforms are skipped
from multiprocessing.pool import ThreadPool
import logging
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import pandas as pd

# you should execute this script as a module to make the following imports work
# (python -m misc_utilities.build_cache)
from training.data.data_api import *
from training.data.load_labels_from_split import load_labels_from_split
from training.data.mia import get_loader as mia_loader


def file_cache_only(dataset: Dataset):
    tfms = []
    for tfm in dataset.transforms:
        tfms.append(tfm)
        if isinstance(tfm, FileCache):
            break

    assert isinstance(tfms[-1], FileCache), "The pipeline does not contain a FileCache"
    return tfms


def build_cache(file_cache: FileCache, inputs, num_workers: int, err_log_path: str):
    def worker(item):
        try:
            file_cache(item)
        except Exception as err:
            print(err)
            return item
        return None

    with ThreadPool(num_workers) as pool:
        err_inputs = list(
            tqdm(
                pool.imap(worker, inputs),
                total=len(inputs),
                desc=file_cache.cache_root.stem[:6],
            )
        )

    err = pd.DataFrame([row for row in err_inputs if row is not None])
    err.to_csv(err_log_path, index=False)


def main(
    data_root,
    cache_root,
    num_workers: int = 20,
    outer_size: int = 256,
    inner_size: int = 224,
):
    logging.basicConfig(level=logging.INFO)

    normal_caches = [
        mia_loader((outer_size, outer_size, outer_size), cache_root),
        mia_loader((inner_size, inner_size, inner_size), cache_root),
    ]

    with logging_redirect_tqdm():
        labels = load_labels_from_split(data_root, patients=None)
        inputs = labels[["patient", "path", "inf", "sev"]].to_dict(orient="records")

        for cache_id, file_cache in enumerate(tqdm(normal_caches, unit="filecache")):
            build_cache(
                file_cache,
                inputs,
                num_workers,
                err_log_path=f"buildcache-error-{cache_id}.csv",
            )


if __name__ == "__main__":
    try:
        import fire

        fire.Fire(main)
    except ModuleNotFoundError:
        # TODO: use argparse
        main()
