from copy import deepcopy
from typing import Callable, List, Optional

from torch.utils.data import DataLoader, DistributedSampler

from chug.common import LoaderBundle, DistributedCfg, SharedCount

from .collate import HfCollate
from .wrappers import SafeDataset, WrappedIterableDataset

_SAMPLE_SHUFFLE_SIZE = 2000


def _disable_decode(ds):
    import datasets

    to_set = []
    for k, v in ds.features.items():
        if isinstance(v, datasets.Image):
            d = deepcopy(v)
            d.decode = False
            to_set.append((k, d))
        elif isinstance(v, datasets.Audio):
            d = deepcopy(v)
            d.decode = False
            to_set.append((k, d))
    for k, d in to_set:
        ds = ds.cast_column(k, d)
    return ds


def create_loader_hf(
        source: str,
        split: str,
        task_pipeline: Optional[List[Callable]] = None,
        data_dir: Optional[str] = None,
        num_samples: Optional[int] = None,
        streaming: bool = False,
        is_training: bool = False,
        batch_size: Optional[int] = 1,
        resampled: bool = False,
        multi_interval: bool = True,
        num_batches_round: str = 'ceil',
        num_workers: int = 4,
        persistent_workers: bool = True,
        start_interval: int = 0,
        seed: int = 0,
        collate_fn: Optional[Callable] = None,
        sample_shuffle_size: int = _SAMPLE_SHUFFLE_SIZE,
        distributed: DistributedCfg = DistributedCfg(),
        disable_decode: bool = False,
):
    """

    Args:
        source:
        split:
        task_pipeline:
        data_dir:
        num_samples:
        streaming:
        is_training:
        batch_size:
        resampled:
        multi_interval:
        num_batches_round:
        num_workers:
        persistent_workers:
        start_interval:
        seed:
        collate_fn:
        sample_shuffle_size:
        distributed:
        disable_decode:

    Returns:

    """
    from datasets import VerificationMode, load_dataset
    batched = batch_size is not None and batch_size >= 0

    if collate_fn is not None:
        assert task_pipeline is None, 'task_pipeline should not be set if custom collation function is used.'
    elif batched or task_pipeline is not None:
        # collation fn applies task pipeline
        assert task_pipeline is not None, 'task_pipeline is needed'
        collate_fn = HfCollate(
            task_pipeline,
            apply_collate=batched,
        )

    if streaming:
        from datasets.distributed import split_dataset_by_node

        dataset = load_dataset(
            source,
            data_dir=data_dir,
            streaming=True,
        )

        if split not in dataset:
            assert False, f'Split {split} not in dataset ({dataset.keys()})'
        dataset = dataset[split]
        if disable_decode:
            dataset = _disable_decode(dataset)

        # FIXME num_samples calc, get a reliable estimate from dataset in streaming mode
        if num_samples is None:
            info = dataset.info
            if info.splits is not None and split in info.splits:
                num_samples = info.splits[split].num_examples

        if is_training and multi_interval:
            assert num_samples, (
                "num_samples must be available in dataset metadata or manually provided for multi-interval training")

        if is_training:
            dataset = dataset.shuffle(seed, buffer_size=sample_shuffle_size)

        # FIXME split_dataset_by_node has some concerns as currently implemented
        dataset = split_dataset_by_node(dataset, distributed.global_rank, distributed.world_size)
        interval_count = SharedCount(start_interval)
        dataset = WrappedIterableDataset(dataset, interval_count=interval_count)

        # HF datasets treats batch_size differently than torch defaults, in torch batch_size = None
        # disables batching, in HF it returns the full dataset. We restore torch behaviour here.
        #batch_size = batch_size or 1
        base_loader = DataLoader(
            dataset=dataset,
            collate_fn=collate_fn,
            sampler=None,
            shuffle=False,
            drop_last=batched and is_training,  # FIXME improve wrt train vs validation vs sharding specifics
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
        )

        batch_size = batch_size or 1
        loader = LoaderBundle(
            loader=base_loader,
            num_batches=num_workers // batch_size,
            num_samples=num_samples,
            shared_interval=interval_count,
        )

    else:
        dataset = load_dataset(
            source,
            data_dir=data_dir,
            verification_mode=VerificationMode.ALL_CHECKS,
        )

        if split not in dataset:
            assert False, f'Split {split} not in dataset ({dataset.keys()})'
        dataset = dataset[split]
        if disable_decode:
            dataset = _disable_decode(dataset)
        dataset = SafeDataset(dataset)

        sampler = None
        if distributed.world_size > 1:
            sampler = DistributedSampler(
                dataset,
                rank=distributed.global_rank,
                shuffle=is_training,
                seed=seed,
                num_replicas=distributed.world_size,
                drop_last=False,
            )
            sampler.set_epoch(start_interval)

        base_loader = DataLoader(
            dataset=dataset,
            collate_fn=collate_fn,
            sampler=sampler,
            drop_last=is_training,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        loader = LoaderBundle(
            loader=base_loader,
            num_batches=len(base_loader),
            num_samples=len(dataset),
            sampler=sampler,
        )

    return loader
