import ast
import json
import os
from typing import Dict

from webdataset.shardlists import expand_urls

from chug.common import SplitInfo


def get_dataset_size(shards):
    shardlist, _ = expand_urls(shards)
    dir_path = os.path.dirname(shardlist[0])

    sizes_filename = os.path.join(dir_path, 'sizes.json')
    len_filename = os.path.join(dir_path, '__len__')

    if os.path.exists(sizes_filename):
        sizes = json.load(open(sizes_filename, 'r'))
        total_size = sum([int(sizes[os.path.basename(shard)]) for shard in shardlist])
    elif os.path.exists(len_filename):
        total_size = ast.literal_eval(open(len_filename, 'r').read())
    else:
        total_size = None  # num samples undefined

    num_shards = len(shardlist)

    return total_size, num_shards

## FIXME this is not working / not completed, parsing _info files is a TODO

def _parse_split_info(split: str, info: Dict):
    def _info_convert(dict_info):
        return SplitInfo(
            num_samples=dict_info['num_samples'],
            filenames=tuple(dict_info['filenames']),
            shard_lengths=tuple(dict_info['shard_lengths']),
            name=dict_info['name'],
        )

    if 'tar' in split or '..' in split:
        split_filenames = expand_urls(split)
        if split_name:
            split_info = info['splits'][split_name]
            if not num_samples:
                _fc = {f: c for f, c in zip(split_info['filenames'], split_info['shard_lengths'])}
                num_samples = sum(_fc[f] for f in split_filenames)
                split_info['filenames'] = tuple(_fc.keys())
                split_info['shard_lengths'] = tuple(_fc.values())
                split_info['num_samples'] = num_samples
            split_info = _info_convert(split_info)
        else:
            split_info = SplitInfo(
                name=split_name,
                num_samples=num_samples,
                filenames=split_filenames,
            )
    else:
        if 'splits' not in info or split not in info['splits']:
            raise RuntimeError(f"split {split} not found in info ({info.get('splits', {}).keys()})")
        split = split
        split_info = info['splits'][split]
        split_info = _info_convert(split_info)

    return split_info