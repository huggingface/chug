from dataclasses import dataclass
from multiprocessing import Value

from torch.utils.data import DataLoader, DistributedSampler


class SharedCount:
    def __init__(self, count: int = 0):
        self.count = Value('i', count)

    def set_value(self, epoch):
        self.count.value = epoch

    def get_value(self):
        return self.count.value


@dataclass
class LoaderBundle:
    loader: DataLoader
    num_batches: int = 0
    num_samples: int = 0
    sampler: DistributedSampler = None
    shared_interval: SharedCount = None

    def set_interval(self, interval):
        if self.shared_interval is not None:
            self.shared_interval.set_value(interval)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(interval)
