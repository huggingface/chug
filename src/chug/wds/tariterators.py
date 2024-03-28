import re

from webdataset.filters import pipelinefilter
from webdataset.tariterators import base_plus_ext, valid_sample, url_opener, tar_file_expander

from .helpers import log_and_continue

BASE_RE = re.compile(r"^((?:.*/|)[^.]+)[.]([^/]*)$")


def base_plus_ext(path):
    """Split off all file extensions.

    Returns base, allext.

    Args:
        path: path with extensions

    Returns:
        path with all extensions removed
    """
    match = re.match(BASE_RE, path)
    if not match:
        return None, None
    return match.group(1), match.group(2)


def group_by_keys_nothrow(
        data,
        keys=base_plus_ext,
        lcase=True,
        suffixes=None,
        handler=None,
):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME wds version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the wds impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


tarfile_to_samples_nothrow = pipelinefilter(tarfile_samples_nothrow)
