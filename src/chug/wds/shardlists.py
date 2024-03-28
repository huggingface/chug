import random
import sys

from torch.utils.data import IterableDataset

from chug.common import SharedCount
from .helpers import expand_urls, pytorch_worker_seed


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
        self.urls = expand_urls(urls)
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


class ResampledShardsV2(IterableDataset):
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
        if weights is not None:
            self.urls, self.weights = expand_urls(urls, weights)
            assert len(self.urls) == len(self.weights), \
                f"Number of urls {len(self.urls)} and weights {len(self.weights)} should match."
        else:
            self.urls = expand_urls(urls)
            self.weights = None
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
