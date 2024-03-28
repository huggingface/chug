from torch.utils.data import Dataset, IterableDataset

from chug.common import SharedCount

class SafeDataset(Dataset):
    """
    This is a Dataset wrapped by a try/except in the __getitem__ in case
    the hfds datasets used have errors/corrupt data.
    """

    def __init__(self, original_dataset, max_retry=10):
        self.ds = original_dataset
        self.max_retry = max_retry

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        err = None
        for try_idx in range(self.max_retry):
            try:
                item = self.ds[idx + try_idx]
                return item
            except Exception as e:
                err = e
                continue
        raise err



class WrappedIterableDataset(IterableDataset):
    """
    """

    def __init__(self, original_dataset, interval_count=None, max_retry=10):
        self.ds = original_dataset
        self.max_retry = max_retry
        self.interval_count = interval_count

    def set_interval_count(self, interval_count):
        if isinstance(self.interval_count, SharedCount):
            self.interval_count.set_value(interval_count)
        else:
            self.interval_count = interval_count

    def __iter__(self):
        if isinstance(self.interval_count, SharedCount):
            interval_count = self.interval_count.get_value()
        else:
            interval_count = self.interval_count
        self.ds.set_epoch(interval_count)
        for sample in self.ds:
            yield sample
