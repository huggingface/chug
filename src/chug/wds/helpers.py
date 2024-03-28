import ast
import json
import logging
import os
import sys

from urllib.parse import urlparse

import braceexpand
import webdataset as wds
from torch.utils.data import get_worker_info


def urldir(url):
    """Return the directory part of a url."""
    parsed_url = urlparse(url)
    path = parsed_url.path
    directory = os.path.dirname(path)
    return parsed_url._replace(path=directory).geturl()


def expand_urls(urls, weights=None):
    if weights is None:
        expanded_urls = wds.shardlists.expand_urls(urls)
        return expanded_urls

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


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True

def dump_and_reraise(exn):
    """Dump stack and stop."""
    import traceback
    exception_trace = ''.join(traceback.format_tb(exn.__traceback__))
    logging.error(f'Handling webdataset {type(exn)}. Exception trace:\n {exception_trace}')
    current_trace = ''.join(traceback.format_tb(exn.__traceback__))
    logging.error(f'Current stack trace:\n {current_trace}')
    raise exn

_error_handlers = {
    'log_and_continue': log_and_continue,
    'ignore_and_continue': wds.ignore_and_continue,
    'warn_and_continue': wds.warn_and_continue,
    'ignore_and_stop': wds.ignore_and_stop,
    'warn_and_stop': wds.warn_and_stop,
    'dump_and_reraise': dump_and_reraise,
    'reraise_exception': wds.reraise_exception,
}

def get_error_handler(name: str):
    return _error_handlers.get(name, dump_and_reraise)
