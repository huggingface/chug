from dataclasses import dataclass
from typing import Callable

import webdataset as wds

from chug.common import DataTaskCfg, FeatureInfo, ImageFeatureInfo
from chug.doc import DocReadProcessor
from chug.wds.helpers import log_and_continue


@dataclass
class DataTaskManualCfg(DataTaskCfg):
    pass


def build_task_pipeline_manual(
        cfg: DataTaskManualCfg,
):
    assert cfg.decode_and_process_fn is not None
    # a pipeline that relies fully on passed in decode_and_process_fn, other cfg fields ignored
    pipe = [
        wds.map(
            cfg.decode_and_process_fn,
            handler=log_and_continue,
        )
    ]
    return pipe
