import random

import numpy as np

from .types import SharedCount


def seed_worker(worker_id):
    import torch
    worker_seed = torch.initial_seed()
    random.seed(worker_seed)
    np.random.seed(worker_seed % 2**32)


def get_pytorch_worker_seed(increment=0, initial_seed=None):
    """get dataloader worker seed from pytorch
    """
    from torch.utils.data import get_worker_info

    increment_value = increment.get_value() if isinstance(increment, SharedCount) else increment
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        num_workers = worker_info.num_workers
        if increment_value:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment_value * max(1, num_workers)
    else:
        # a fallback when no dataloader workers are present (num_workers=0)
        import torch

        if initial_seed is None:
            initial_seed = torch.initial_seed()

        # generate seed from initial via torch.Generator so it matches DL worker seeds
        seed = torch.empty((), dtype=torch.int64).random_(
            generator=torch.Generator().manual_seed(initial_seed)).item()

        if increment_value:
            seed += increment_value

    return seed
