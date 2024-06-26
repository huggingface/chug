from .pipeline_doc_read import build_task_pipeline_doc_read, DataTaskDocReadCfg
from .pipeline_doc_vqa import build_task_pipeline_doc_vqa, DataTaskDocVqaCfg
from .pipeline_gtparse import build_task_pipeline_gtparse
from .pipeline_image_text import build_task_pipeline_image_text, DataTaskImageTextCfg
from .pipeline_manual import build_task_pipeline_manual, DataTaskManualCfg

from.pipeline_factory import create_task_pipeline