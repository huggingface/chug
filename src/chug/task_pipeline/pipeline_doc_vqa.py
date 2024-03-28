from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Dict, Optional, Union

import webdataset as wds

from chug.common import DataTaskCfg, FeatureInfo, ImageFeatureInfo
from chug.wds import get_error_handler
from chug.doc import (
    DocVqaProcessor,
    DEFAULT_DOC_KEY,
    DEFAULT_QUESTION_KEY,
    DEFAULT_QUESTION_ID_KEY,
    DEFAULT_ANSWER_KEY,
    DEFAULT_DOC_KEY_TUPLE,
    DEFAULT_QUESTION_KEY_TUPLE,
    DEFAULT_ANSWER_KEY_TUPLE,
    DEFAULT_DOC_FEAT,
    DEFAULT_QUESTION_FEAT,
    DEFAULT_QUESTION_ID_FEAT,
    DEFAULT_ANSWER_FEAT
)


def filter_missing(
        sample,
        image_key=DEFAULT_DOC_KEY_TUPLE,
        question_key=DEFAULT_QUESTION_KEY_TUPLE,
        answer_key=DEFAULT_ANSWER_KEY_TUPLE,
):
    has_question = any(k in sample for k in question_key)
    has_answer = any(k in sample for k in answer_key)
    has_image = any(k in sample for k in image_key)
    return has_question and has_answer and has_image


# Currently assuming this schema as default, one set of question/answers per image, images possibly duplicated
# sample = {
#     'png': bytes,
#     'question_id': 33,
#     'doc_id': 55,  # optional
#     'question': 'what is a trumpet?',
#     'answers': ['an instrument', 'a brass instrument']
# }
#


@dataclass
class DataTaskDocVqaCfg(DataTaskCfg):
    """
    Attributes:
        answer_feat:
        question_feat:
        question_id_feat:
        image_input_feat:
        text_input_feat:
        text_target_feat:
        question_prefix:
        question_suffix:
        answer_prefix:
        answer_suffix:
        render_dpi:
    """
    answer_feat: FeatureInfo = DEFAULT_ANSWER_FEAT
    question_feat: FeatureInfo = DEFAULT_QUESTION_FEAT
    question_id_feat: FeatureInfo = DEFAULT_QUESTION_ID_FEAT
    image_input_feat: ImageFeatureInfo = DEFAULT_DOC_FEAT
    text_input_feat: FeatureInfo = FeatureInfo('text_input', input_key=None)
    text_target_feat: FeatureInfo = FeatureInfo('text_target', input_key=None)
    question_prefix: Optional[str] = '<s_question>'
    question_suffix: Optional[str] = '</s_question>'
    answer_prefix: Optional[str] = '<s_answer>'
    answer_suffix: Optional[str] = '</s_answer>'
    # FIXME prompt templates instead of prefix+suffix above?
    render_dpi: int = 144


def build_task_pipeline_doc_vqa(
        cfg: DataTaskDocVqaCfg,
):
    # document decoding & pre-processing done together, there is coupling in random page
    # selection and in the future, masking of image and/or text
    handler = get_error_handler(cfg.error_handler)

    pipe = [
        wds.map(
            DocVqaProcessor(
                image_process_fn=cfg.image_process_fn,
                text_process_fn=cfg.text_process_fn,
                image_input_feat=cfg.image_input_feat,
                question_feat=cfg.question_feat,
                answer_feat=cfg.answer_feat,
                question_id_feat=cfg.question_id_feat,
                render_dpi=cfg.render_dpi,
                question_prefix=cfg.question_prefix,
                question_suffix=cfg.question_suffix,
                answer_prefix=cfg.answer_prefix,
                answer_suffix=cfg.answer_suffix,
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

