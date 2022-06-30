import logging
from torch.utils.data import DataLoader


class ZipLoader:
    def __init__(self, main_loader: DataLoader, other_loader: DataLoader):
        self.main_loader = main_loader
        self.other_loader = other_loader

    @property
    def dataset(self):
        return self.main_loader.dataset

    def __iter__(self):
        return _ZipLoaderIter(self.main_loader, self.other_loader)

    def __len__(self):
        return len(self.main_loader) + len(self.other_loader)


class _ZipLoaderIter:
    def __init__(self, main_loader: DataLoader, other_loader: DataLoader):
        self.main_loader = main_loader
        self.other_loader = other_loader
        self.main_iter = iter(main_loader)
        self.other_iter = iter(other_loader)
        self._main_returned = 0
        self._other_returned = 0

    def __next__(self):
        main_ratio = self._main_returned / len(self.main_loader)
        other_ratio = self._other_returned / len(self.other_loader)

        if main_ratio >= 1 and other_ratio >= 1:
            raise StopIteration

        if main_ratio < other_ratio:
            self._main_returned += 1
            return next(self.main_iter)
        else:
            self._other_returned += 1
            return next(self.other_iter)
