import dataclasses
import math
from typing import Callable, List, Optional, Union

import webdataset as wds

from chug.common import DistributedCfg, LoaderBundle, SharedCount, ShardSpec
from .helpers import expand_urls
from .pipeline import build_data_pipeline


_SAMPLE_SHUFFLE_SIZE = 2000
_SAMPLE_SHUFFLE_INITIAL = 500


def create_loader_wds(
        shards: Union[str, List[str], ShardSpec],
        task_pipeline: Optional[List[Callable]],
        num_samples: Optional[int] = None,
        is_training: bool = False,
        batch_size: Optional[int] = 1,
        resampled: bool = False,
        multi_interval: bool = True,
        num_batches_round: str = 'ceil',
        num_workers: int = 4,
        persistent_workers: bool = True,
        start_interval: int = 0,
        seed: int = 0,
        handler: Callable = wds.reraise_exception,
        collate_fn: Optional[Callable] = None,
        sample_shuffle_size: int = _SAMPLE_SHUFFLE_SIZE,
        sample_shuffle_initial: int = _SAMPLE_SHUFFLE_INITIAL,
        distributed: DistributedCfg = DistributedCfg(),
):
    """ Create a webdataset loader

    Args:
        shards:
        task_pipeline:
        is_training:
        resampled: If True, shards are resampled with replacement.
        multi_interval: If True, run loader in multi-interval mode (multi-epoch), num_samples is interpreted
            as num_samples per interval (epoch if # samples == dataset length). Dataset length is set to a
            fixed value to approximate at-least once sample visiting per interval.
            If False, num_samples is treated as total samples to visit over training without paying attention
            to # samples in underlying dataset. Dataset length is not accessible.
        start_interval:
        seed:
        num_workers:
        persistent_workers:
        batch_size:
        num_batches_round:
        collate_fn:
        sample_shuffle_size:
        sample_shuffle_initial:
        distributed:

    Returns:

    """
    resampled = resampled and is_training

    if not isinstance(shards, ShardSpec):
        if isinstance(shards, str):
            shards = expand_urls(shards)
        shards = ShardSpec(
            urls=shards,
        )

    num_shards = len(shards.urls)
    if num_samples is None:
        if shards.sizes:
            num_samples = sum(shards.sizes)
        if is_training and not num_samples:
            raise RuntimeError(
                'The number of dataset samples must be specified for the training dataset '
                'if no dataset length info is present.')

    num_batches_per_worker = 0
    if is_training:
        assert batch_size >= 1, 'batching must be enabled for train, set batch_size>=1'
        num_workers = max(1, num_workers)
        # We want to see the same # of batches on each member of the distributed group (GPU),
        # this is enforced by making each worker produce the same # of batches regardless of the
        # underlying iterator, so we estimate and make the iterator wrap around if end is hit.
        # This will repeat some samples and may miss some sample per interval as shards may be
        # uneven or allocated unevenly across all workers. There are ways improve on this naive
        # approach, to get closer to the ideal of each sample in an interval (epoch) seen once,
        # but difficult to achieve perfectly, and most improvements require full co-ordination across
        # all workers via out-of-band IPC/RPC.

        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if num_batches_round == 'floor' else math.ceil
        global_batch_size = batch_size * distributed.world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_batches_per_worker = round_fn(num_batches / num_workers)  # per dataloader worker
        num_batches = num_batches_per_worker * num_workers
        num_samples = num_batches * global_batch_size
    else:
        # Eval / inference will exhaust the iterator if the size is not specified.
        # Eval currently supported for 1 train process only (primary)
        # FIXME support distributed eval
        #  https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel.join
        num_samples = num_samples or 0
        # last batches are partial, eval is done on a single (primary) process
        if batch_size:
            num_batches = math.ceil(num_samples / batch_size)
        else:
            num_batches = 0

    if is_training:
        # create a shared epoch store to sync epoch to dataloader worker proc
        shared_interval_count = SharedCount(count=start_interval)
        if not resampled:
            assert num_shards >= num_workers * distributed.world_size, 'number of shards must be >= total workers'
    else:
        shared_interval_count = None

    datapipe = build_data_pipeline(
        shards=shards,
        task_pipeline=task_pipeline,
        is_training=is_training,
        batch_size=batch_size,
        resampled=resampled,
        multi_interval=multi_interval,
        seed=seed,
        shared_interval_count=shared_interval_count,
        num_batches_per_worker=num_batches_per_worker,
        sample_shuffle_initial=sample_shuffle_initial,
        sample_shuffle_size=sample_shuffle_size,
        handler=handler,
        collate_fn=collate_fn,
    )

    dataloader = wds.WebLoader(
        datapipe,
        batch_size=None,  # batching done in data-pipeline
        shuffle=False,  # shuffling done in data-pipeline
        num_workers=num_workers,
        persistent_workers=persistent_workers and num_workers > 0,
    )

    return LoaderBundle(
        loader=dataloader,
        num_batches=num_batches,
        num_samples=num_samples,
        shared_interval=shared_interval_count,
    )


