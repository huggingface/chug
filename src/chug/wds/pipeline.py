from typing import Callable, List, Optional, Sequence, Union

import torch.utils.data
import webdataset
import webdataset as wds

from chug.common import ShardSpec, collate
from .helpers import log_and_continue
from .shardlists import ResampledShardsV2, ShuffledShardList
from .tariterators import tarfile_to_samples_nothrow


def build_data_pipeline(
        shards: ShardSpec,
        task_pipeline: List[Callable],
        is_training: bool = False,
        batch_size: int = 0,
        resampled: bool = False,
        multi_interval: bool = False,
        seed: int = 0,
        shared_interval_count: int = -1,
        num_batches_per_worker: int = 0,
        sample_shuffle_initial: int = 1,
        sample_shuffle_size: int = 1,
        collate_fn: Optional[Callable] = None,
        batched_task_pipeline: Optional[List[Callable]] = None,
        handler=wds.reraise_exception,
):
    """

    Args:
        task_pipeline: task-specific pipeline for processing samples
        shards: ShardSpec w/ shard url list and optional sizes and weights
        is_training: Train mode. Enables shuffling of samples and forces consistent batch #s across workers.
        batch_size: Batch for sample collation. Collation / batching disabled if batch_size == 0.
        resampled: Enable resampling with replacement of shards.
        multi_interval:
        seed:
        shared_interval_count:
        num_batches_per_worker:
        sample_shuffle_initial:
        sample_shuffle_size:
        collate_fn:
        batched_task_pipeline: An optional task-specific pipeline for processing batched samples
        handler: Exception handler

    Returns:

    """
    if not isinstance(task_pipeline, (list, tuple)):
        task_pipeline = [task_pipeline]
    assert len(task_pipeline)

    if resampled:
        datapipe = [ResampledShardsV2(
            shards.urls,
            weights=shards.weights,
            deterministic=True,
            interval=shared_interval_count,
        )]
    else:
        assert shards.weights is None, \
            "upsampling_factors is only supported when sampling with replacement (resampled=False)."
        if is_training:
            datapipe = [ShuffledShardList(
                shards.urls,
                seed=seed,
                interval=shared_interval_count,
            )]
        else:
            datapipe = [wds.SimpleShardList(
                shards.urls,
            )]

    # at this point we have an iterator over all the shards
    if is_training:
        if not resampled:
            datapipe.extend([
                wds.split_by_node,
                wds.split_by_worker,
            ])
        # at this point, we have an iterator over the shards assigned to each worker at each node
        datapipe.extend([
            tarfile_to_samples_nothrow(handler=handler),
            wds.shuffle(
                bufsize=sample_shuffle_size,
                initial=sample_shuffle_initial,
            ),
        ])
    else:
        datapipe.extend([
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(handler=handler),
        ])

    # task specific decode and map pipline (per-sample)
    datapipe.extend(task_pipeline)

    # collation, tensor output disabled with batch_size == 0 or None
    if batch_size:
        # NOTE torch default_collate handles dicts, wds default_collate does not
        collate_fn = collate_fn or collate
        datapipe.extend([
            wds.batched(
                batch_size,
                partial=not is_training,
                collation_fn=collate_fn,
            )
        ])

        # task-specific batched pipeline (per-batch)
        if batched_task_pipeline:
            datapipe.extend(batched_task_pipeline)

    datapipe = wds.DataPipeline(*datapipe)

    if is_training and num_batches_per_worker > 0:
        if multi_interval:
            datapipe = datapipe.with_epoch(num_batches_per_worker)  # each worker is iterating over this
        else:
            datapipe = datapipe.repeat(nbatches=num_batches_per_worker)

    return datapipe
