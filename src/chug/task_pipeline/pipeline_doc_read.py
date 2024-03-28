from dataclasses import dataclass, field
from typing import Callable, Optional

import webdataset as wds

from chug.common import DataTaskCfg, FeatureInfo, ImageFeatureInfo
from chug.doc import DocReadProcessor, DEFAULT_DOC_FEAT
from chug.wds import get_error_handler


@dataclass
class DataTaskDocReadCfg(DataTaskCfg):
    image_input_feat: ImageFeatureInfo = DEFAULT_DOC_FEAT
    text_input_feat: FeatureInfo = FeatureInfo('text_input', input_key='pages')
    text_target_feat: Optional[FeatureInfo] = FeatureInfo('text_target', input_key=None)
    page_sampling: str = 'random'
    render_dpi: int = 150


def build_task_pipeline_doc_read(
        cfg: DataTaskDocReadCfg,
):
    handler = get_error_handler(cfg.error_handler)
    pipe = []

    # document decoding & pre-processing done together, there is coupling in random page
    # selection and in the future, masking of image and/or text
    pipe += [
        wds.map(
            DocReadProcessor(
                image_process_fn=cfg.image_process_fn,
                text_process_fn=cfg.text_process_fn,
                image_input_feat=cfg.image_input_feat,
                text_input_feat=cfg.text_input_feat,
                text_target_feat=cfg.text_target_feat,
                page_sampling=cfg.page_sampling,
                render_dpi=cfg.render_dpi,
                flatten_json=cfg.flatten_json,
            ),
            handler=handler,
        )
    ]

    if cfg.output_tuple:
        # NOTE in this mode we lose '_parse' key and would need to derive from target
        # Unless we add support for parse as the last tuple element?
        if cfg.text_target_feat is not None:
            pipe += [
                wds.to_tuple(
                    cfg.image_input_feat.output_name,
                    cfg.text_input_feat.output_name,
                    cfg.text_target_feat.output_name,
                )
            ]
        else:
            pipe += [
                wds.to_tuple(
                    cfg.image_input_feat.output_name,
                    cfg.text_input_feat.output_name,
                )
            ]
    return pipe
