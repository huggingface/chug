from dataclasses import asdict
from typing import Any, Dict, Optional, Union

from torchvision import transforms
from timm.data import (
    ResizeKeepRatio,
    CenterCropOrPad,
    RandomResizedCropAndInterpolation,
    create_transform,
)

from chug.common import ImageInputCfg, ImageAugCfg
from .transforms_torch import ConvertColor


def build_transforms_image_timm(
        input_cfg: ImageInputCfg,
        is_training: bool = False,
        do_normalize: bool = True,
        do_convert: bool = False,
        aug_cfg: Optional[Union[Dict[str, Any], ImageAugCfg]] = None,
        composed: bool = True,
):
    """ Build image transforms leveraging timm's create_transform() functionality.

    Args:
        input_cfg:
        is_training: In training mode, apply train transforms w/ augmentations
        do_normalize: Enable normalization of ouput tensors by specified mean & std deviation.
        aug_cfg:

    Returns:

    """
    interpolation = input_cfg.interpolation or 'bicubic'
    assert interpolation in ['bicubic', 'bilinear', 'random']

    resize_mode = input_cfg.resize_mode or 'shortest'
    assert resize_mode in ('shortest', 'longest', 'squash')

    if isinstance(aug_cfg, dict):
        aug_cfg = ImageAugCfg(**aug_cfg)
    else:
        aug_cfg = aug_cfg or ImageAugCfg.imagenet()

    if is_training:
        aug_cfg_dict = {k: v for k, v in asdict(aug_cfg).items() if v is not None}
        aug_cfg_dict.setdefault('color_jitter', None)  # disable by default
        # FIXME map aug_cfg -> timm.create_transform args more carefully

        train_transform = create_transform(
            input_size=input_cfg.size,
            is_training=True,
            use_prefetcher=not do_normalize,  # FIXME prefetcher mode disables normalize, but outputs np.array
            hflip=0.,
            mean=input_cfg.mean,
            std=input_cfg.std,
            re_mode='pixel',
            interpolation=interpolation,
            **aug_cfg_dict,
        )
        return train_transform
    else:
        if resize_mode == 'longest':
            timm_crop_mode = 'border'
        elif resize_mode == 'squash':
            timm_crop_mode = 'squash'
        else:
            assert resize_mode == 'shortest'
            timm_crop_mode = 'center'

        eval_transform = create_transform(
            input_size=input_cfg.size,
            is_training=False,
            use_prefetcher=not do_normalize,  # FIXME prefetcher mode disables normalize, but outputs np.array
            mean=input_cfg.mean,
            std=input_cfg.std,
            crop_pct=1.0,
            crop_mode=timm_crop_mode,
            # FIXME
            # composed=composed,
        )
        return eval_transform


def build_transforms_image_basic(
        input_cfg: ImageInputCfg,
        is_training: bool = False,
        do_normalize: bool = True,
        do_convert: bool = False,
        aug_cfg: Optional[Union[Dict[str, Any], ImageAugCfg]] = None,
        composed: bool = True,
):
    """ Build image transfoms leveraging torchvision transforms.
    """
    if do_normalize:
        normalize = transforms.Normalize(mean=input_cfg.mean, std=input_cfg.std)
    else:
        normalize = None

    interpolation = input_cfg.interpolation or 'bicubic'
    assert interpolation in ['bicubic', 'bilinear', 'random']
    # NOTE random is ignored for interpolation_mode, so defaults to BICUBIC for inference if set
    if interpolation == 'bilinear':
        interpolation_mode = transforms.InterpolationMode.BILINEAR
    else:
        interpolation_mode = transforms.InterpolationMode.BICUBIC

    resize_mode = input_cfg.resize_mode or 'shortest'
    assert resize_mode in ('shortest', 'longest', 'squash')

    if isinstance(aug_cfg, dict):
        aug_cfg = ImageAugCfg(**aug_cfg)
    else:
        aug_cfg = aug_cfg or ImageAugCfg.imagenet()

    if is_training:
        image_size = input_cfg.size
        if resize_mode == 'shortest':
            if isinstance(image_size, (tuple, list)) and image_size[0] == image_size[1]:
                image_size = image_size[0]  # w/ scalar for final resize in RRC will use shortest edge
            # FIXME note we don't have good option for 'longest' resizing w/ RRC

        transform_list = [
            # like torchvision.transforms.RandomResizedCrop but supports randomized interpolation for robustness
            RandomResizedCropAndInterpolation(
                image_size,
                scale=aug_cfg.scale or (1.0, 1.0),
                ratio=aug_cfg.ratio or (1.0, 1.0),
                interpolation=interpolation,
            ),
        ]

        if do_convert:
            transform_list.append(ConvertColor(mode=input_cfg.mode))

        if aug_cfg.color_jitter_prob:
            assert aug_cfg.color_jitter is not None and len(aug_cfg.color_jitter) == 4
            if aug_cfg.color_jitter_prob is not None:
                transform_list.append(
                    transforms.RandomApply(
                        transforms.ColorJitter(
                            *aug_cfg.color_jitter,
                        ),
                        p=aug_cfg.color_jitter_prob,
                    )
                )
        elif aug_cfg.color_jitter is not None:
            transform_list.append(transforms.ColorJitter(*aug_cfg.color_jitter))

        if aug_cfg.grayscale_prob:
            transform_list.append(transforms.RamndomGrayscale(aug_cfg.grayscale_prob))

        if aug_cfg.gaussian_blur_prob:
            gaussian_blur_kernel = aug_cfg.gaussian_blur_kernel_size or 23
            transforms.RandomApply([
                transforms.GaussianBlur(
                    kernel_size=gaussian_blur_kernel,
                )],
                p=aug_cfg.gaussian_blur_prob,
            )

    else:
        image_size = input_cfg.size

        if resize_mode == 'longest':
            transform_list = [
                ResizeKeepRatio(image_size, interpolation=interpolation_mode, longest=1),
                CenterCropOrPad(image_size, fill=input_cfg.fill_color)
            ]
        elif resize_mode == 'squash':
            if isinstance(image_size, int):
                image_size = (image_size, image_size)

            transform_list = [
                transforms.Resize(image_size, interpolation=interpolation_mode, antialias=True),
            ]
        else:
            assert resize_mode == 'shortest'
            if not isinstance(image_size, (tuple, list)):
                image_size = (image_size, image_size)

            if image_size[0] == image_size[1]:
                # simple case, use torchvision built-in Resize w/ shortest edge mode (scalar size arg)
                transform_list = [
                    transforms.Resize(image_size[0], interpolation=interpolation_mode, antialias=True)
                ]
            else:
                # resize shortest edge to matching target dim for non-square target
                transform_list = [ResizeKeepRatio(image_size)]

            transform_list += [transforms.CenterCrop(image_size)]

        if do_convert:
            transform_list.append(ConvertColor(mode=input_cfg.mode))
    # end if is_training

    transform_list += [transforms.ToTensor()]

    if normalize is not None:
        transform_list += [normalize]

    return transforms.Compose(transform_list) if composed else transform_list
