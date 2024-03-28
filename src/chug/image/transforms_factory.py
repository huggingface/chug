from typing import Any, Dict, Optional, Union

from chug.common import ImageInputCfg, ImageAugCfg
from .build_transforms_doc import build_transforms_doc_better, build_transforms_doc_nougat, build_transforms_doc_basic
from .build_transforms_image import build_transforms_image_timm, build_transforms_image_basic

_transform_factories = {
    "image_basic": build_transforms_image_basic,
    "image_timm": build_transforms_image_timm,
    "doc_basic": build_transforms_doc_basic,
    "doc_nougat": build_transforms_doc_nougat,
    "doc_better": build_transforms_doc_better,
}

def build_image_transforms(
        input_cfg: ImageInputCfg,
        is_training=True,
        do_normalize=True,
        do_convert=False,
        composed=True,
        aug_cfg: Optional[Union[Dict[str, Any], ImageAugCfg]] = None,
):
    common_args = dict(
        input_cfg=input_cfg,
        is_training=is_training,
        do_normalize=do_normalize,
        aug_cfg=aug_cfg,
        composed=composed,
    )

    tt = input_cfg.transform_type
    assert tt in _transform_factories, \
        f"Unrecognized transform type: {tt}. Must be one of {list(_transform_factories.keys())}."
    transforms = _transform_factories[tt](**common_args)

    return transforms


def create_image_preprocessor(
        input_cfg: ImageInputCfg,
        is_training=True,
        do_normalize=True,
        do_convert=False,
        aug_cfg: Optional[Union[Dict[str, Any], ImageAugCfg]] = None,
):
    transforms = build_image_transforms(
        input_cfg=input_cfg,
        is_training=is_training,
        do_normalize=do_normalize,
        do_convert=do_convert,
        aug_cfg=aug_cfg,
        composed=True,
    )
    # NOTE for now, a stack of composed transforms are the image pre-processor
    return transforms

