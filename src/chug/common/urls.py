import os
from numbers import Number
from typing import Sequence


import braceexpand
import re


def envlookup(m):
    """Look up match in the environment with prefix WDS_.

    Args:
        m: a match object

    Returns:
        str: the value of the environment variable WDS_<m.group(1)>
    """
    key = m.group(1)
    for prefix in ('WDS_', 'CHUG_'):
        key = prefix + key
        if key in os.environ:
            return os.environ[key]
    assert key in os.environ, f"missing WDS/CHUG environment variable for {key}"


def envsubst(s):
    """Substitute ${var} with the value of the environment variable WDS_var.

    Args:
        s (str): string to be substituted

    Returns:
        str: the substituted string
    """
    return re.sub(r"\$\{(\w+)\}", envlookup, s)


def _subst_and_expand(url: str):
    for i in range(10):
        last = url
        url = envsubst(url)
        if url == last:
            break
    return braceexpand.braceexpand(url)


def expand_urls(urls, weights=None):
    """ Expand urls (and optionally weights) if they are strings, otherwise return as lists.
    """
    if weights is None:
        if isinstance(urls, str):
            url_list = urls.split("::")
            result = []
            for url in url_list:
                result.extend(_subst_and_expand(url))
            return result, None
        else:
            return list(urls), None

    if isinstance(urls, str):
        url_list = urls.split('::')

        if isinstance(weights, str):
            weights = weights.split('::')
        elif isinstance(weights, Number):
            weights = [weights] * len(url_list)
        assert len(weights) == len(url_list), \
            f"Expected the number of data components ({len(url_list)}) and weights({len(weights)}) to match."
        weights = [float(weight) for weight in weights]
        all_urls, all_weights = [], []
        for url, weight in zip(url_list, weights):
            expanded_url = list(_subst_and_expand(url))
            expanded_weights = [weight] * len(expanded_url)
            all_urls.extend(expanded_url)
            all_weights.extend(expanded_weights)
    else:
        all_urls = list(urls)
        if isinstance(weights, Number):
            # if weights is a scalar, expand to url list
            all_weights = [float(weights)] * len(all_urls)
        else:
            assert len(weights) == len(all_urls), \
                f"Expected the number of data components ({len(all_urls)}) and weights({len(weights)}) to match."
            all_weights = list(weights)

    return all_urls, all_weights

