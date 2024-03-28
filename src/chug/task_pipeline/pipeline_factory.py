from chug.common import DataTaskCfg

from .pipeline_doc_read import build_task_pipeline_doc_read, DataTaskDocReadCfg
from .pipeline_doc_vqa import build_task_pipeline_doc_vqa, DataTaskDocVqaCfg
from .pipeline_image_text import build_task_pipeline_image_text, DataTaskImageTextCfg
# from .donut_gtparse_pipe import
from .pipeline_manual import build_task_pipeline_manual, DataTaskManualCfg

_cfg_to_create = {
    DataTaskDocReadCfg: build_task_pipeline_doc_read,
    DataTaskDocVqaCfg: build_task_pipeline_doc_vqa,
    DataTaskImageTextCfg: build_task_pipeline_image_text,
    DataTaskManualCfg: build_task_pipeline_manual,
}


def create_task_pipeline(cfg: DataTaskCfg):
    create_fn = _cfg_to_create[type(cfg)]
    return create_fn(cfg)
