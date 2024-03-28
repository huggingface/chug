from typing import Callable, List, Optional

from chug.common import DataCfg, DataTaskCfg, DistributedCfg, LoaderBundle, source_to_shard_spec, SourceSpec
from chug.hfds import create_loader_hf
from chug.task_pipeline import create_task_pipeline
from chug.wds import create_loader_wds, get_error_handler


def create_loader(
        data_cfg: DataCfg,
        task_cfg: DataTaskCfg,
        task_pipeline: Optional[List[Callable]] = None,
        is_training: bool = False,
        start_interval: int = 0,
        seed: int = 0,
        distributed: DistributedCfg = DistributedCfg(),
) -> LoaderBundle:
    """
    Creates a dataloader for training or validation based on configuration settings.

    Parameters:
        data_cfg: Configuration object for the dataset.
        task_cfg: Configuration object for the task specific processing.
        task_pipeline: Task specific processing pipeline (takes priority over task_cfg).
        is_training : Indicates if the loader is for training data (True) or validation data (False).
        start_interval: The starting interval (epoch for full passes) for setting seed, etc. appropriately.
        seed: Seed for random operations to ensure reproducibility.
        distributed: Distributed device information.

    Returns:
        DataLoader: A PyTorch DataLoader instance configured according to the provided settings.

    Note:
        Currently supports "wds" and "hf_dataset" as dataset formats.
    """
    if data_cfg.format == "wds":
        loader = create_loader_from_config_wds(
            data_cfg=data_cfg,
            task_cfg=task_cfg,
            task_pipeline=task_pipeline,
            is_training=is_training,
            start_interval=start_interval,
            seed=seed,
            distributed=distributed,
        )

    elif data_cfg.format.startswith("hf"):
        loader = create_loader_from_config_hf(
            data_cfg=data_cfg,
            task_cfg=task_cfg,
            task_pipeline=task_pipeline,
            is_training=is_training,
            start_interval=start_interval,
            seed=seed,
            distributed=distributed,
        )

    else:
        assert False, f"Unsupported dataset format ({data_cfg.format})."

    return loader


def _validate_cfgs(
    data_cfg: DataCfg,
    task_cfg: Optional[DataTaskCfg],
    is_training: bool = False,
):
    batch_size = data_cfg.batch_size
    if batch_size is not None:
        if task_cfg.decode_and_process_fn is None:
            # FIXME make validation task specific once we have tasks that don't require both image and text preproc
            assert task_cfg.image_process_fn is not None and task_cfg.text_process_fn is not None,\
                'task_cfg.image_process_fn and task_cfg.text_process_fn must be set if batching enabled'
        else:
            assert task_cfg.decode_fn is None, \
                'task_cfg.decode_fn should not be set at the same time as task_cfg.decode_and_process_fn'


def create_loader_from_config_wds(
        data_cfg: DataCfg,
        task_cfg: Optional[DataTaskCfg],
        task_pipeline: Optional[List[Callable]] = None,
        is_training: bool = False,
        start_interval: int = 0,
        seed: int = 0,
        collate_fn: Optional[Callable] = None,
        distributed: DistributedCfg = DistributedCfg(),
):
    """

    Args:
        data_cfg:
        task_cfg:
        task_pipeline:
        is_training:
        start_interval:
        seed:
        collate_fn:
        distributed:

    Returns:

    """
    _validate_cfgs(data_cfg, task_cfg, is_training=is_training)

    if task_pipeline is None:
        assert task_cfg is not None
        task_pipeline = create_task_pipeline(
            task_cfg,
        )

    handler = get_error_handler(task_cfg.error_handler)

    return create_loader_wds(
        shards=data_cfg.shard_spec,
        task_pipeline=task_pipeline,
        num_samples=data_cfg.num_samples,
        is_training=is_training,
        resampled=data_cfg.resampled,
        multi_interval=True, # FIXME via config?
        num_workers=data_cfg.num_workers,
        batch_size=data_cfg.batch_size,
        persistent_workers=data_cfg.persistent_workers,
        collate_fn=collate_fn,
        start_interval=start_interval,
        seed=seed,
        handler=handler,
        distributed=distributed,
    )


def create_loader_from_config_hf(
        data_cfg: DataCfg,
        task_cfg: DataTaskCfg,
        task_pipeline: Optional[List[Callable]] = None,
        is_training: bool = False,
        start_interval: int = 0,
        seed: int = 0,
        distributed: DistributedCfg = DistributedCfg(),
):
    """

    Args:
        data_cfg:
        task_cfg:
        task_pipeline:
        is_training:
        start_interval:
        seed:
        distributed:

    Returns:

    """
    assert not isinstance(data_cfg.source, (list, tuple)), "Multiple sources not supported for HF datasets."
    assert isinstance(data_cfg.source, (str, SourceSpec)), \
        "The specified source for HF dataset must be a string or SourceSpec."

    _validate_cfgs(data_cfg, task_cfg, is_training=is_training)

    if isinstance(data_cfg.source, SourceSpec):
        source = data_cfg.source.url
        split = data_cfg.source.split
        assert split, "Split must be set in SourceSpec with HF datasets."
    else:
        source = data_cfg.source
        split = data_cfg.split
        assert split, "Split must be set in DataCfg when string source is used with HF datasets."

    if task_pipeline is None:
        assert task_cfg is not None
        task_pipeline = create_task_pipeline(
            task_cfg,
        )

    streaming = 'hfids' in data_cfg.format

    return create_loader_hf(
        source=source,
        split=split,
        task_pipeline=task_pipeline,
        streaming=streaming,
        is_training=is_training,
        batch_size=data_cfg.batch_size,
        data_dir=data_cfg.data_dir,
        num_samples=data_cfg.num_samples,
        num_workers=data_cfg.num_workers,
        persistent_workers=data_cfg.persistent_workers,
        seed=seed,
        distributed=distributed,
    )
