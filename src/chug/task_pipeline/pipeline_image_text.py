from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional, Union

import webdataset as wds

from chug.common import DataTaskCfg, ImageFeatureInfo, FeatureInfo
from chug.wds import get_error_handler, create_image_decoder

_DEFAULT_IMG_KEY = "jpg;png;jpeg;webp;tif"
_DEFAULT_TXT_KEY = "txt"
_DEFAULT_IMG_KEY_TUPLE = tuple(_DEFAULT_IMG_KEY.split(';'))
_DEFAULT_TXT_KEY_TUPLE = tuple(_DEFAULT_TXT_KEY.split(';'))


@dataclass
class DataTaskImageTextCfg(DataTaskCfg):
    image_input_feat: ImageFeatureInfo = ImageFeatureInfo()
    text_input_feat: FeatureInfo = FeatureInfo('text', input_key=_DEFAULT_TXT_KEY)


def filter_incomplete(
        sample,
        image_key=_DEFAULT_IMG_KEY_TUPLE,
        text_key=_DEFAULT_TXT_KEY_TUPLE
):
    has_caption = any(k in sample for k in text_key)
    has_image = any(k in sample for k in image_key)
    return has_caption and has_image


def build_task_pipeline_image_text(
        cfg: DataTaskImageTextCfg,
):
    """ Create pipeline for dual image & text input pipelines.
    """
    handler = get_error_handler(cfg.error_handler)
    pipe = []

    # FIXME add support for caption target for caption tasks or use a separate pipe?

    if cfg.filter_valid:
        filter_fn = partial(
            filter_incomplete,
            image_key=tuple(cfg.image_input_feat.input_key.split(';')),
            text_key=tuple(cfg.text_input_feat.input_key.split(';')),
        )
        pipe += [
            wds.select(filter_fn)
        ]

    if cfg.decode_and_process_fn:
        pipe += [
            wds.map(cfg.decode_and_process_fn)
        ]
    else:
        decode_fn = create_image_decoder(
            cfg.decode_fn,
            image_mode=cfg.image_input_feat.image_mode,
            handler=handler,
        )

        rename_dict = {
            cfg.image_input_feat.output_name: cfg.image_input_feat.input_key,
            cfg.text_input_feat.output_name: cfg.text_input_feat.input_key,
        }
        pipe += [
            decode_fn,
            wds.rename(**rename_dict, keep=False, handler=handler),
        ]

        map_dict = {}
        if cfg.image_process_fn is not None:
            map_dict[cfg.image_input_feat.output_name] = cfg.image_process_fn
        if cfg.text_process_fn is not None:
            map_dict[cfg.text_input_feat.output_name] = cfg.text_process_fn
        if map_dict:
            pipe += [
                wds.map_dict(**map_dict, handler=handler)
            ]

    if cfg.output_tuple:
        pipe += [
            wds.to_tuple(
                cfg.image_input_feat.output_name,
                cfg.text_input_feat.output_name,
            )
        ]

    return pipe
