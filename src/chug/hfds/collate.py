from typing import List, Callable, Optional

import torch.utils
from torch.utils.data import IterableDataset, DataLoader

from chug.common import collate


def invoke(f, *args, **kwargs):
    if isinstance(f, (IterableDataset, DataLoader)) and len(args) == 0:
        return iter(f)
    if isinstance(f, list):
        return iter(f)
    if callable(f):
        result = f(*args, **kwargs)
        return result
    raise ValueError(f"{f}: not a valid pipeline stage")


"""
pipe = [wds.rename(image='jpg'), wds.map_dict(image=tf), wds.to_tuple('image', 'cls')]
hfc = HfCollate(pipe)
dl = DataLoader(ds, batch_size=32, num_workers=4, persistent_workers=True, collate_fn=hfc)
"""

def flatten_bytes(data):
    for sample in data:
        to_replace = {k for k, v in sample.items() if isinstance(v, dict) and 'bytes' in v}
        if to_replace:
            result = {k: v for k, v in sample.items() if k not in to_replace}
            result.update({k: sample[k]['bytes'] for k in to_replace})
            yield result
        else:
            yield sample


class HfCollate:
    """ Collation wrapper that applies processing pipeline for HF datasets use
    """
    def __init__(
        self,
            pipeline: List[Callable],
            collate_fn: Optional[Callable] = None,
            apply_collate: bool = True,
    ):
        """
        Args:
            pipeline: list of pipeline functions
            collate_fn: use a custom collation function, otherwise defaults to torch default_collate
        """
        self.pipeline = pipeline
        self.collate_fn = collate_fn or collate
        self.apply_collate = apply_collate
        self._debug = False

    def __call__(self, batch):
        item = False
        if not self.apply_collate and isinstance(batch, dict):
            batch = [batch]
            item = True

        if self._debug:
            for b in batch:
                for k, v in b.items():
                    print(k, type(v))
                    if isinstance(v, torch.Tensor):
                        print(v.shape)

        if self.pipeline:
            for pipe_fn in self.pipeline:
                batch = invoke(pipe_fn, batch)

        batch = list(batch)

        if self._debug:
            for b in batch:
                for k, v in b.items():
                    print(k, v)

        if self.apply_collate:
            return self.collate_fn(batch)
        else:
            return batch[0] if item else batch
