import itertools
import random
from typing import Mapping, Sequence, Union

import webdataset as wds
from webdataset.filters import _shuffle

from chug.common import SharedCount, get_pytorch_worker_seed


class detshuffle_v2(wds.PipelineStage):

    def __init__(
            self,
            bufsize: int = 1000,
            initial: int = 100,
            seed: int = 0,
            interval: Union[int, SharedCount] =-1,
            unique_worker: bool = False,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.interval = interval
        self.unique_worker = unique_worker

    def run(self, src):
        if isinstance(self.interval, SharedCount):
            interval = self.interval.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.interval += 1
            interval = self.interval

        rng = random.Random()
        if self.unique_worker:
            # Use the PyTorch worker's seed, *different* across all nodes/workers
            # but also deterministic if they are set consistently
            seed = get_pytorch_worker_seed(interval, initial_seed=self.seed)
        else:
            # This seed to be deterministic AND the *same* across all nodes/workers in each epoch/interval
            seed = self.seed + interval
        rng.seed(seed)

        return _shuffle(src, self.bufsize, self.initial, rng)


def _map_v2(data, f, handler=wds.reraise_exception):
    """ Map samples.

    This function differs from wds.map, it only adds '__key__' back to sample if it exists.

    """
    for sample in data:
        try:
            result = f(sample)
        except Exception as exn:
            if handler(exn):
                continue
            else:
                break
        if result is None:
            continue
        if isinstance(sample, dict) and isinstance(result, dict) and "__key__" in sample:
            result["__key__"] = sample.get("__key__")
        yield result


map_v2 = wds.pipelinefilter(_map_v2)


def _expand_maybe(data, f, handler=wds.reraise_exception):
    for sample in data:
        if isinstance(sample, Mapping):
            try:
                result = f(sample)
            except Exception as exn:
                if handler(exn):
                    continue
                else:
                    break
            if result is None:
                continue
            if "__key__" in sample:
                result["__key__"] = sample["__key__"]
            yield result
        else:
            assert isinstance(sample, Sequence)
            for subsample in sample:
                assert isinstance(subsample, Mapping)
                try:
                    result = f(subsample)
                except Exception as exn:
                    if handler(exn):
                        continue
                    else:
                        break
                if result is None:
                    continue
                if "__key__" in subsample:
                    result["__key__"] = subsample["__key__"]
                yield result


map_expand_maybe = wds.pipelinefilter(_expand_maybe)


def _expand_always(data, f, handler=wds.reraise_exception):
    for sample in itertools.chain(*data):
        assert isinstance(sample, Mapping)
        try:
            result = f(sample)
        except Exception as exn:
            if handler(exn):
                continue
            else:
                break
        if result is None:
            continue
        if "__key__" in sample:
            result["__key__"] = sample["__key__"]
        yield result


map_expand_always = wds.pipelinefilter(_expand_always)


def _flatten_nested(data, *args, replace_existing=True, remove_original=True):
    """Convert dict samples to tuples."""
    for sample in data:
        for k in args:
            nested_dict = sample.pop(k, {}) if remove_original else sample.get(k, {})
            if replace_existing:
                sample.update(nested_dict)
            elif k in sample:
                for sk, sv in nested_dict.items():
                    sample.setdefault(sk, sv)
        yield sample


flatten_nested = wds.pipelinefilter(_flatten_nested)