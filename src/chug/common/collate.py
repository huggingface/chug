import collections
from typing import Callable, Dict, Optional, Tuple, Type, Union

from torch.utils.data._utils.collate import default_collate_fn_map, default_collate_err_msg_format


def collate(batch):
    r"""
        A customized collate function that handles collection type of element within each batch.

        This collate function has been tweaked to provide different functionality when handling
        dictionary samples. Certain keys are excluded or not tensorized.

        Args:
            batch: a single batch to be collated
    """
    elem = batch[0]
    elem_type = type(elem)

    if elem_type in default_collate_fn_map:
        return default_collate_fn_map[elem_type](batch)

    for collate_type in default_collate_fn_map:
        if isinstance(elem, collate_type):
            return default_collate_fn_map[collate_type](batch)

    if isinstance(elem, collections.abc.Mapping):
        try:
            out = {}
            for key in elem:
                if key.startswith('__'):
                    # skip keys starting with '__', e.g. '__key__',
                    continue
                elif key.startswith('_'):
                    # do not recurse or tensorize values for keys starting with '_', e.g. '_parse'
                    out[key] = [d[key] for d in batch]
                else:
                    out[key] =  collate([d[key] for d in batch])
            out = elem_type(out)
            return out
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return {key: collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.

        if isinstance(elem, tuple):
            return [collate(samples) for samples in transposed]  # Backwards compatibility.
        else:
            try:
                return elem_type([collate(samples) for samples in transposed])
            except TypeError:
                # The sequence type may not support `__init__(iterable)` (e.g., `range`).
                return [collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))