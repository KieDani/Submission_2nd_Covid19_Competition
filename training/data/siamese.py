import logging
from torch.utils.data import DataLoader 
from training.data.data_api import Dataset


class SiameseLoader:
    """A wrapper around two PyTorch data loaders. One for positive and one for negative
    samples.
    """
    def __init__(
        self,
        dataset: Dataset,
        num_workers: int,
        batch_size: int,
        shuffle: bool = False,
        worker_init_fn=None,
    ):
        pos_input = [inp for inp in dataset.input if inp["sev"] == 1]
        neg_input = [inp for inp in dataset.input if inp["sev"] == 0]
        assert len(pos_input) + len(neg_input) == len(dataset)

        if not shuffle and len(pos_input) != len(neg_input):
            logging.warn(
                f"No shuffling and unequal amount of pos/neg samples! pos: {len(pos_input)}, neg: {len(neg_input)}"
            )

        pos_data = Dataset(pos_input, dataset.transforms)
        neg_data = Dataset(neg_input, dataset.transforms)

        self.pos_loader = DataLoader(
            dataset=pos_data,
            num_workers=num_workers,
            batch_size=batch_size,
            shuffle=shuffle,
            worker_init_fn=worker_init_fn,
        )
        self.neg_loader = DataLoader(
            dataset=neg_data,
            num_workers=num_workers,
            batch_size=batch_size,
            shuffle=shuffle,
            worker_init_fn=worker_init_fn,
        )

    @property
    def dataset(self):
        return self.pos_loader.dataset

    def __iter__(self):
        return zip(self.pos_loader, self.neg_loader)

    def __len__(self):
        return min(len(self.pos_loader), len(self.neg_loader))
