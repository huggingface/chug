from dataclasses import dataclass, field, fields, replace
from typing import Any, Dict, List, Optional, Tuple, Union, Sequence, Callable

from simple_parsing.helpers import Serializable

from .types import ImageFeatureInfo, ShardSpec, SourceSpec
from .urls import expand_urls


def image_mode_to_chs(fmt: str):
    if fmt is None:
        return None
    assert fmt in ('L', 'RGB')   # could support more...
    return 1 if fmt == 'L' else 3


@dataclass
class ImageInputCfg(Serializable):
    size: Optional[Tuple[int, int]] = (512, 512)
    mode: Optional[str] = 'L'
    mean: Optional[Union[float, Tuple[float, ...]]] = 0.5
    std: Optional[Union[float, Tuple[float, ...]]] = 0.5
    interpolation: Optional[str] = 'bicubic'
    fill_color: Optional[Union[int, Tuple[int, ...]]] = 255
    crop_margin: Optional[bool] = False
    align_long_axis: Optional[bool] = False
    transform_type: Optional[str] = 'image_basic'
    resize_mode: Optional[str] = 'shortest'

    @property
    def image_chs(self):
        return image_mode_to_chs(self.mode)

    def __post_init__(self):
        image_chs = self.image_chs
        if image_chs is not None:
            # ensue mean/std attributes match # image_chs
            for attr_name in ('mean', 'std'):
                attr = getattr(self, attr_name)
                if attr is not None and not isinstance(attr, Sequence):
                    attr = (attr,)
                if image_chs == 1 and len(attr) > image_chs:
                    attr = (sum(attr) / len(attr),)
                if image_chs > 1 and len(attr) == 1:
                    attr = attr * image_chs
                assert len(attr) == image_chs
                setattr(self, attr_name, attr)

            # ensure fill color matches image_chs
            if self.fill_color is not None:
                if not isinstance(self.fill_color, Sequence):
                    self.fill_color = (self.fill_color,)
                if image_chs == 1 and len(self.fill_color) > image_chs:
                    self.fill_color = (int(sum(self.fill_color) / len(self.fill_color)),)
                if image_chs > 1 and len(self.fill_color) == 1:
                    self.fill_color  = self.fill_color * image_chs

    @classmethod
    def empty(cls):
        return cls(**{f.name: None for f in fields(cls)})

    def set_default(self, right, inplace=False):
        # set left fields from right fields if left fields are not-initialized (None)
        changes = {
            f.name: v for f in fields(self)
            if (v := getattr(right, f.name)) is not None and getattr(self, f.name) is None
        }
        if inplace:
            for k, v in changes.items():
                setattr(self, k, v)
            return self
        else:
            return replace(self, **changes)

    def merge(self, right, inplace=False):
        # merge from right to left for right fields that are not None
        changes = {f.name: v for f in fields(self) if (v := getattr(right, f.name)) is not None}
        if inplace:
            for k, v in changes.items():
                setattr(self, k, v)
            return self
        else:
            return replace(self, **changes)


@dataclass
class ImageAugCfg(Serializable):
    """
    A simple flat config struct for overriding common augmentation defaults.

    Each image transform type supports different augmentations and have their own defaults,
    this struct is intended to override defaults for common values, not necessarily to
    cover all cases and define all augmentation possibilities across all schemes.
    """

    # resize scale bounds (1.0 = middle point = same scale)
    scale: Optional[Tuple[float, float]] = None

    # resize aspect ratio bounds (1.0 = 1:1)
    ratio: Optional[Tuple[float, float]] = None

    # color jitter, per item probs
    color_jitter: Optional[Union[float, Tuple[float, float, float], Tuple[float, float, float, float]]] = None

    # for simclr, control prob for applying any of the jitter probs above
    color_jitter_prob: Optional[float] = None

    # for preprocess w/ grayscale (simclr), control prob of converting to graysacle
    grayscale_prob: Optional[float] = None

    gaussian_blur_prob: Optional[float] = None
    gaussian_blur_kernel_size: Optional[int] = None

    # probability of applying random-erasing (timm style aug)
    re_prob: Optional[float] = None

    # number of random-erasing blocks (timm style aug)
    re_count: Optional[int] = None

    @classmethod
    def clip(cls, **kwargs):
        aug = cls(
            scale=(0.9, 1.0),
            ratio=(0.75, 1. / 0.75),
        )
        aug = replace(aug, **kwargs)
        return aug

    @classmethod
    def imagenet(cls, **kwargs):
        aug = cls(
            scale=(0.08, 1.0),
            ratio=(0.75, 1. / 0.75),
            color_jitter=(0.4, 0.4, 0.4),
        )
        aug = replace(aug, **kwargs)
        return aug

    @classmethod
    def simclr(cls, **kwargs):
        aug = cls(
            scale=(0.08, 1.0),
            ratio=(0.75, 1. / 0.75),
            color_jitter=(0.4, 0.4, 0.4, 0.1),
            color_jitter_prob=0.8,
            grayscale_prob=0.2,
            gaussian_blur_prob=0.5,
            #gaussian_blur_kernel_size=23,
        )
        aug = replace(aug, **kwargs)
        return aug


# Vision preprocessing config
@dataclass
class PreprocessCfg(Serializable):
    image_input: ImageInputCfg = field(default_factory=ImageInputCfg)
    aug_cfg: Optional[ImageAugCfg] = None


@dataclass
class DataArg(Serializable):
    """ Data source argument in an argument friendly form (multiple sources represented in string)
    """
    source: str
    split: Optional[str] = None
    sampling_weight: Optional[str] = None
    template: Optional[str] = None  # template to transform url for use
    num_samples: Optional[Union[int, str]] = None
    data_dir: Optional[str] = None

    batch_size: int = 1
    format: str = "wds"  # e.g. "hfds", "hfids", or "wds"

    resampled: bool = False  # sample shards with replacement
    multi_interval: bool = True
    persistent_workers: bool = True
    num_workers: int = 4


def split_sources(
        source: str,
        split: Optional[str] = None,
        sampling_weights: Optional[Union[str, List[float]]] = None,
        num_samples: Optional[Union[int, str, List[int]]] = None,
):
    if '::' in source:
        source_split = source.split('::')
    else:
        source_split = [source]
    num_sources = len(source_split)

    if sampling_weights is not None:
        if isinstance(sampling_weights, str):
            weights_split = sampling_weights.split('::')
            sampling_weights = [float(w) for w in weights_split]
        assert len(sampling_weights) == num_sources

    num_samples_per_source = None
    if num_samples is not None:
        if isinstance(num_samples, str):
            num_samples_split = num_samples.split('::')
            num_samples = [int(s) for s in num_samples_split]

        try:
            len(num_samples)
        except Exception:
            num_samples_per_source = [None] * num_sources
        else:
            num_samples_per_source = num_samples
            num_samples = sum(num_samples_per_source)
        finally:
            assert len(num_samples_per_source) == num_sources

    output = []
    for i, s in enumerate(source_split):
        output.append(SourceSpec(
            url=s,
            split=split,
            sampling_weight=None if sampling_weights is None else sampling_weights[i],
            num_samples=None if num_samples_per_source is None else num_samples_per_source[i],
        ))

    return output, num_samples


# FIXME add code to resolve shard information from _info.yaml or .json files (see dataset_info.py)


def source_to_shard_spec(
        source: Union[str, SourceSpec, List[SourceSpec]],
):
    if isinstance(source, str):
        source_list = [SourceSpec(url=source)]
    elif isinstance(source, SourceSpec):
        source_list = [source]
    else:
        assert isinstance(source[0], SourceSpec)
        source_list = source

    # process weights first in case some are set and some are not
    if not all(s.sampling_weight is None for s in source_list):
        weights = [s.sampling_weight if s.sampling_weight else 1.0 for s in source_list]
    else:
        weights = [None] * len(source_list)

    all_urls = []
    all_weights = []
    for s, w in zip(source_list, weights):
        expanded_urls, expanded_weights = expand_urls(s.url, weights=w)
        all_urls.extend(expanded_urls)
        if expanded_weights:
            all_weights.extend(expanded_weights)
    all_weights = all_weights or None
    sizes = None  # FIXME resolve sizes

    ss = ShardSpec(urls=all_urls, weights=all_weights, sizes=sizes)
    return ss


@dataclass
class DataCfg(Serializable):
    source: Union[str, SourceSpec, List[SourceSpec]]
    split: Optional[str] = None
    num_samples: Optional[int] = None  # overrides num_samples across sources if set
    data_dir: Optional[str] = None

    batch_size: Optional[int] = 1
    format: str = "wds"  # e.g. "hfds", "hfids", or "wds".

    resampled: bool = False  # sample shards with replacement
    multi_interval: bool = True
    persistent_workers: bool = True
    num_workers: int = 4

    @classmethod
    def from_arg(cls, data_arg: DataArg):
        sources, _ = split_sources(
            data_arg.source,
            data_arg.num_samples,
            data_arg.num_samples,
        )
        return cls(
            source=sources,
            num_samples=data_arg.num_samples,
            data_dir=data_arg.data_dir,
            batch_size=data_arg.batch_size,
            format=data_arg.format,
            resampled=data_arg.resampled,
            multi_interval=data_arg.multi_interval,
            persistent_workers=data_arg.persistent_workers,
            num_workers=data_arg.num_workers,
        )

    @property
    def shard_spec(self):
        return source_to_shard_spec(self.source)

    def __post_init__(self):
        if self.num_workers == 0:
            self.persistent_workers = False

@dataclass
class DistributedCfg:
    world_size: int = 1
    local_rank: int = 0
    global_rank: int = 0
