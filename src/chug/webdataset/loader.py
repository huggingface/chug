import ast
import json
import logging
import math
import os
import random
import sys
from typing import Callable, List, Optional, Union

from torch.utils.data import IterableDataset, get_worker_info

import braceexpand
import webdataset as wds
from webdataset.filters import _shuffle
from webdataset.tariterators import base_plus_ext, url_opener, tar_file_expander, valid_sample

from chug.common import LoaderBundle, SharedCount

_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


def expand_urls(urls, weights=None):
    if weights is None:
        expanded_urls = wds.shardlists.expand_urls(urls)
        return expanded_urls, None

    if isinstance(urls, str):
        urllist = urls.split("::")
        weights = weights.split('::')
        assert len(weights) == len(urllist), \
            f"Expected the number of data components ({len(urllist)}) and weights({len(weights)}) to match."
        weights = [float(weight) for weight in weights]
        all_urls, all_weights = [], []
        for url, weight in zip(urllist, weights):
            expanded_url = list(braceexpand.braceexpand(url))
            expanded_weights = [weight for _ in expanded_url]
            all_urls.extend(expanded_url)
            all_weights.extend(expanded_weights)
        return all_urls, all_weights
    else:
        all_urls = list(urls)
        return all_urls, weights


def get_dataset_size(shards):
    shards_list, _ = expand_urls(shards)
    dir_path = os.path.dirname(shards_list[0])
    sizes_filename = os.path.join(dir_path, 'sizes.json')
    len_filename = os.path.join(dir_path, '__len__')
    if os.path.exists(sizes_filename):
        sizes = json.load(open(sizes_filename, 'r'))
        total_size = sum([int(sizes[os.path.basename(shard)]) for shard in shards_list])
    elif os.path.exists(len_filename):
        total_size = ast.literal_eval(open(len_filename, 'r').read())
    else:
        total_size = None  # num samples undefined
        # some common dataset sizes (at time of authors last download)
        # CC3M (train): 2905954
        # CC12M: 10968539
        # LAION-400M: 407332084
        # LAION-2B (english): 2170337258
    num_shards = len(shards_list)
    return total_size, num_shards


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True


def group_by_keys_nothrow(
        data,
        keys=base_plus_ext,
        lcase=True,
        suffixes=None,
        handler=None,
):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def pytorch_worker_seed(increment=0):
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


class detshuffle2(wds.PipelineStage):

    def __init__(
            self,
            bufsize=1000,
            initial=100,
            seed=0,
            interval=-1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.interval = interval

    def run(self, src):
        if isinstance(self.interval, SharedCount):
            interval = self.interval.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.interval += 1
            interval = self.interval
        rng = random.Random()
        if self.seed < 0:
            # If seed is negative, we use the worker's seed, this will be different across all nodes/workers
            seed = pytorch_worker_seed(interval)
        else:
            # This seed to be deterministic AND the same across all nodes/workers in each epoch
            seed = self.seed + interval
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


class ShuffledShardList(IterableDataset):
    """An iterable dataset yielding a list of urls that is deterministically shuffled based on epoch."""

    def __init__(
            self,
            urls,
            seed=0,
            interval=-1,
            num_sub_intervals=None,
    ):
        """Iterate through the list of shards."""
        super().__init__()
        urls, _ = expand_urls(urls)
        self.urls = urls
        assert len(self.urls) and isinstance(self.urls[0], str)
        self.seed = seed
        self.interval = interval
        self.num_sub_intervals = num_sub_intervals  # FIXME experimental feature

    def __len__(self):
        return len(self.urls)

    def __iter__(self):
        """Return an iterator over the shards."""
        urls = self.urls.copy()

        # Set epoch
        if isinstance(self.interval, SharedCount):
            interval = self.interval.get_value()
        else:
            # NOTE: this is interval tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.interval += 1
            interval = self.interval

        if self.seed is not None:
            # Shuffle with the same seed across all nodes/workers in each interval or super interval
            if self.num_sub_intervals is None:
                seed = self.seed + interval
            else:
                # Keep shuffling consistent across the super epochs
                seed = self.seed + (interval // self.num_sub_intervals)
            random.Random(seed).shuffle(urls)

        # Restrict to shards in the sub epoch if needed
        if self.num_sub_intervals is not None:
            urls = urls[interval % self.num_sub_intervals::self.num_sub_intervals]

        # Yield shards
        for url in urls:
            yield dict(url=url)


class ResampledShards2(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(
            self,
            urls,
            weights=None,
            nshards=sys.maxsize,
            worker_seed_fn=None,
            deterministic=False,
            interval=-1,
    ):
        """Sample shards from the shard list with replacement.

        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        urls, weights = expand_urls(urls, weights)
        self.urls = urls
        self.weights = weights
        if self.weights is not None:
            assert len(self.urls) == len(self.weights), \
                f"Number of urls {len(self.urls)} and weights {len(self.weights)} should match."
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed_fn = worker_seed_fn
        self.deterministic = deterministic
        self.interval = interval

    def __iter__(self):
        """Return an iterator over the shards."""
        if isinstance(self.interval, SharedCount):
            interval = self.interval.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.interval += 1
            interval = self.interval

        if self.deterministic:
            # reset seed w/ interval if deterministic
            if self.worker_seed_fn is None:
                # pytorch worker seed should be deterministic (per-worker)
                # It is init by arg.seed + rank + worker id
                seed = pytorch_worker_seed(interval)
            else:
                seed = self.worker_seed_fn() + interval
            self.rng.seed(seed)

        for _ in range(self.nshards):
            if self.weights is None:
                yield dict(url=self.rng.choice(self.urls))
            else:
                yield dict(url=self.rng.choices(self.urls, weights=self.weights, k=1)[0])


def create_wds_loader(
        data_spec: Union[str, List[str]],
        decoder_pipeline: List[Callable],
        is_train: bool = False,
        resampled: bool = False,
        multi_interval=True,
        start_interval=0,
        num_samples: Optional[int] = None,
        seed: int = 0,
        workers: int = 4,
        persistent_workers: bool = True,
        batch_size: int = 256,
        world_size: int = 1,
        upsampling_factors: List[float] = None,
        num_batches_round='ceil',
        datapipe_only=False,
        sample_shuffle_size=_SAMPLE_SHUFFLE_SIZE,
        sample_shuffle_initial=_SAMPLE_SHUFFLE_INITIAL,
):
    """

    Args:
        data_spec:
        decoder_pipeline:
        is_train:
        resampled: If True, shards are resampled with replacement.
        multi_interval: If True, run loader in multi-interval mode (multi-epoch), num_samples is interpreted
            as num_samples per interval (epoch if # samples == dataset length). Dataset length is set to a
            fixed value to approximate at-least once sample visiting per interval.
            If False, num_samples is treated as total samples to visit over training without paying attention
            to # samples in underlying dataset. Dataset length is not accessible.
        start_interval:
        num_samples:
        seed:
        workers:
        persistent_workers:
        batch_size:
        world_size:
        upsampling_factors:
        num_batches_round:

        datapipe_only:

    Returns:

    """
    assert data_spec is not None
    resampled = resampled and is_train

    num_shards = None
    if is_train:
        if not multi_interval:
            assert num_samples, "num_samples to be seen over training must be specified"

        if num_samples is not None:
            num_samples = num_samples
        else:
            num_samples, num_shards = get_dataset_size(data_spec)
            if not num_samples:
                raise RuntimeError(
                    'The number of dataset samples must be specified for the training dataset '
                    'if no dataset length info is present.')
    else:
        # Eval will just exhaust the iterator if the size is not specified.
        # Eval currently supported for 1 train process only (primary)
        # FIXME support distributed eval
        num_samples = num_samples or 0

    if is_train:
        # create a shared epoch store to sync epoch to dataloader worker proc
        shared_interval_count = SharedCount(count=start_interval)
    else:
        shared_interval_count = None

    if resampled:
        datapipe = [ResampledShards2(
            data_spec,
            weights=upsampling_factors,
            deterministic=True,
            interval=shared_interval_count,
        )]
    else:
        assert upsampling_factors is None, \
            "upsampling_factors is only supported when sampling with replacement (resampled=False)."
        if is_train:
            datapipe = [ShuffledShardList(
                data_spec,
                seed=seed,
                interval=shared_interval_count,
            )]
        else:
            datapipe = [wds.SimpleShardList(
                data_spec,
            )]

    # at this point we have an iterator over all the shards
    if is_train:
        if not resampled:
            datapipe.extend([
                wds.split_by_node,
                wds.split_by_worker,
            ])
        # at this point, we have an iterator over the shards assigned to each worker at each node
        datapipe.extend([
            tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
            wds.shuffle(
                bufsize=sample_shuffle_size,
                initial=sample_shuffle_initial,
            ),
        ])
    else:
        datapipe.extend([
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(handler=log_and_continue),
        ])
    datapipe.extend(decoder_pipeline)
    datapipe.extend([
        wds.batched(batch_size, partial=not is_train)
    ])

    datapipe = wds.DataPipeline(*datapipe)

    if is_train:
        num_workers = max(1, workers)
        # We want to see the same # of batches on each member of the distributed group (GPU),
        # this is enforced by making each worker produce the same # of batches regardless of the
        # underlying iterator, so we estimate and make the iterator wrap around if end is hit.
        # This will repeat some samples and may miss some sample per interval as shards may be
        # uneven or allocated unevenly across all workers. There are ways improve on this naive
        # approach, to get closer to the ideal of each sample in an interval (epoch) seen once,
        # but difficult to achieve perfectly, and most improvements require full co-ordination across
        # all workers via out-of-band IPC/RPC.
        if not resampled:
            num_shards = num_shards or len(expand_urls(data_spec)[0])
            assert num_shards >= num_workers * world_size, 'number of shards must be >= total workers'

        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if num_batches_round == 'floor' else math.ceil
        global_batch_size = batch_size * world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        if multi_interval:
            datapipe = datapipe.with_epoch(num_worker_batches)  # each worker is iterating over this
        else:
            datapipe = datapipe.repeat(nbatches=num_worker_batches)
    else:
        # last batches are partial, eval is done on a single (primary) process
        # FIXME support distributed eval via
        #  https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel.join
        num_batches = math.ceil(num_samples / batch_size)

    if datapipe_only:
        return datapipe

    dataloader = wds.WebLoader(
        datapipe,
        batch_size=None,
        shuffle=False,
        num_workers=workers,
        persistent_workers=persistent_workers and workers > 0,
    )

    return LoaderBundle(
        loader=dataloader,
        num_batches=num_batches,
        num_samples=num_samples,
        shared_interval=shared_interval_count,
    )
