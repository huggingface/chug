from dataclasses import dataclass
from multiprocessing import Value
from typing import Any, Dict, List, Optional, Tuple, Union
from numbers import Number

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
    """
    Bundle a DataLoader with num_batch / num_sample limits, sampler or shared_interval counter exposed
    to allow easy seed control per-interval.
    """
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

    def __iter__(self):
        return self.loader.__iter__()


@dataclass
class SplitInfo:
    filenames: Tuple[str]
    num_samples: int
    shard_lengths: Tuple[int] = ()
    name: str = ''


# @dataclass
# class ShardInfo:
#     url: str
#     weight: float = 1.0
#     num_samples: Optional[int] = None
#
#


@dataclass
class SourceSpec:
    url: str
    split: Optional[str] = None  # dataset split
    template: Optional[str] = None  # template to transform url -> usage
    sampling_weight: Optional[float] = None
    num_samples: Optional[int] = None

    # TODO resolve dataset info and track base url, shard info (sizes, etc)
    # base_url: str = None
    # info_url: str = None


@dataclass
class SourceInfo(SourceSpec):
    split_info: Dict[str, SplitInfo] = None
    shard_info: Dict[str, Dict[str, Any]] = None


@dataclass
class ShardSpec:
    urls: List[str]
    weights: Optional[Union[float, List[float]]] = None
    sizes: Optional[List[int]] = None

    def __post_init__(self):
        num_shards = len(self.urls)
        if self.weights is not None:
            if isinstance(self.weights, Number):
                self.weights = [self.weights] * num_shards
            assert len(self.weights) == num_shards
        if self.sizes is not None:
            assert len(self.sizes) == num_shards


@dataclass(frozen=True)
class FeatureInfo:
    """ Feature Information

    Attributes:
        output_name: output feature name, None if an intermediary feature
        input_key: input dataset key(s), ';' delimited for multiple options
    """
    output_name: Optional[str] = 'image'
    input_key: Optional[str] = 'jpg;png'
    #parent: Optional[str] = None


@dataclass(frozen=True)
class ImageFeatureInfo(FeatureInfo):
    """ Image Feature Information

    Attributes:
        image_mode: Image colour mode (e.g. 'RGB', 'RGBA', 'L')
        output_name: output feature name, None if an intermediary feature
        input_key: input dataset key(s), ';' delimited for multiple options
        parent: parent key to search for input_key, e.g. 'json'
    """
    image_mode: str = 'RGB'
