from typing import Optional

from chug.common import ImageInputCfg, ImageAugCfg

from .transforms_torch import AlignLongAxis, Bitmap, Erosion, Dilation, CropMargin

# NOTE, chug currently depends on some time aug impl, this should be flipped if timm ends up
# leveraging chug data pipelines.
from timm.data import str_to_interp_mode, ResizeKeepRatio, CenterCropOrPad, RandomCropOrPad

from torchvision import transforms


def build_transforms_doc_basic(
        input_cfg: ImageInputCfg,
        is_training: bool = False,
        do_normalize: bool = True,
        aug_cfg: Optional[ImageAugCfg] = None,
        composed: bool = True,
):
    # an improved torchvision + custom op transforms (no albumentations)
    image_size = input_cfg.size
    interpolation_mode = str_to_interp_mode(input_cfg.interpolation)

    pp = []

    if input_cfg.crop_margin:
        pp += [CropMargin()]

    if input_cfg.align_long_axis:
        pp += [AlignLongAxis(image_size, interpolation=interpolation_mode)]

    if is_training:
        pp += [
            RandomCropOrPad(image_size, fill=input_cfg.fill_color),
            transforms.CenterCrop(image_size),
        ]
    else:
        pp += [
            ResizeKeepRatio(image_size, longest=1, interpolation=input_cfg.interpolation),
            CenterCropOrPad(image_size, fill=input_cfg.fill_color),
        ]

    pp += [transforms.ToTensor()]

    if do_normalize:
        pp += [transforms.Normalize(input_cfg.mean, input_cfg.std)]

    return transforms.Compose(pp) if composed else pp


def build_transforms_doc_better(
        input_cfg: ImageInputCfg,
        is_training: bool = False,
        do_normalize: bool = True,
        aug_cfg: Optional[ImageAugCfg] = None,
        composed: bool = True,
):
    # an improved torchvision + custom op transforms (no albumentations)
    image_size = input_cfg.size
    interpolation_mode = str_to_interp_mode(input_cfg.interpolation)
    pp = []

    if input_cfg.crop_margin:
        pp += [CropMargin()]

    if input_cfg.align_long_axis:
        pp += [AlignLongAxis(image_size, interpolation=interpolation_mode)]

    if is_training:
        # FIXME merge defaults w/ aug_cfg
        defaults = dict(
            scale_prob=0.05,
            scale_range=(0.85, 1.04),
            ratio_prob=0.05,
            ratio_range=(.9, 1.11),
            bitmap_prob=0.55,
            erosion_dilation_prob=0.02,
            shear_prob=0.05,
            shear_range_x=(0, 3.),
            shear_range_y=(-3, 0),
            shift_scale_rotate_prob=0.03,
            shift_range_x=0.04,
            shift_range_y=0.03,
            rotate_range=3,
            elastic_prob=0.04,
            elastic_alpha=50.,
            elastic_sigma=12.,
            brightness_contrast_prob=0.04,
            brightness_range=0.1,
            contrast_range=0.1,
            gaussian_blur_prob=0.03,
            gaussian_blur_kernel=3,
        )
        params = defaults

        pp += [
            ResizeKeepRatio(
                image_size,
                longest=1,
                interpolation=input_cfg.interpolation,
                random_scale_prob=params['scale_prob'],
                random_scale_range=params['scale_range'],
                random_aspect_prob=params['ratio_prob'],
                random_aspect_range=params['ratio_range'],
            ),
            transforms.RandomApply([
                Bitmap()
                ],
                p=params['bitmap_prob']
            ),
            transforms.RandomApply([
                transforms.RandomChoice([
                    Erosion(3),
                    Dilation(3),
                ])],
                p=params['erosion_dilation_prob']
            ),
            transforms.RandomApply([
                transforms.RandomAffine(
                    degrees=0,
                    shear=params['shear_range_x'] + params['shear_range_y'],
                    interpolation=interpolation_mode,
                    fill=input_cfg.fill_color,
                )],
                p=params['shear_prob'],
            ),
            transforms.RandomApply([
                transforms.RandomAffine(
                    degrees=params['ratio_range'],
                    translate=(params['shift_range_x'], params['shift_range_y']),
                    interpolation=interpolation_mode,
                    fill=input_cfg.fill_color,
                )],
                p=params['shift_scale_rotate_prob'],
            ),
            transforms.RandomApply([
                transforms.ElasticTransform(
                    alpha=params['elastic_alpha'],
                    sigma=params['elastic_sigma'],
                    interpolation=interpolation_mode,
                    fill=input_cfg.fill_color,
                )],
                p=params['elastic_prob'],
            ),
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=params['brightness_range'],
                    contrast=params['contrast_range'],
                )],
                p=params['brightness_contrast_prob'],
            ),
            transforms.RandomApply([
                transforms.GaussianBlur(
                    params['gaussian_blur_kernel'],
                    sigma=(0.1, 0.8),
                )],
                p=params['gaussian_blur_prob'],
            ),
            RandomCropOrPad(image_size, fill=input_cfg.fill_color),
            transforms.CenterCrop(image_size),
        ]
    else:
        pp += [
            ResizeKeepRatio(image_size, longest=1, interpolation=input_cfg.interpolation),
            CenterCropOrPad(image_size, fill=input_cfg.fill_color),
        ]

    pp += [transforms.ToTensor()]

    if do_normalize:
        pp += [transforms.Normalize(input_cfg.mean, input_cfg.std)]

    return transforms.Compose(pp) if composed else pp


def build_transforms_doc_nougat(
        input_cfg: ImageInputCfg,
        is_training: bool = False,
        do_normalize: bool = True,
        aug_cfg: Optional[ImageAugCfg] = None,
        composed: bool = True,
):
    import albumentations as alb
    from chug.image.transforms_alb import BitmapAlb, ErosionAlb, DilationAlb, AlbWrapper, CropMarginCv2

    # albumentations + custom opencv transforms from nougat
    image_size = input_cfg.size
    if input_cfg.interpolation == 'bilinear':
        interpolation_mode = 1
    else:
        interpolation_mode = 2  # bicubic
    border_mode = 0

    tv_pp = []
    alb_pp = []

    if input_cfg.crop_margin:
        tv_pp += [CropMarginCv2()]

    if input_cfg.align_long_axis:
        tv_pp += [AlignLongAxis(image_size)]

    if is_training:
        # FIXME merge defaults w/ aug_cfg
        defaults = dict(
            #scale_prob=0.05,
            scale_range=(0.85, 1.03), # done as part of shift_scale_rotate
            #ratio_prob=0.05,
            #ratio_range=(.9, 1.11),
            bitmap_prob=0.05,
            erosion_dilation_prob=0.02,
            erosion_dilation_scale=(2, 3),
            shear_prob=0.03,
            shear_range_x=(0, 3.),
            shear_range_y=(-3, 0),
            shift_scale_rotate_prob=0.03,
            shift_range_x=(0, 0.04),
            shift_range_y=(0, 0.03),
            rotate_range=2.,
            grid_distort_prob=0.04,
            grid_distort_range=0.05,
            elastic_prob=0.04,
            elastic_alpha=50.,
            elastic_sigma=12.,
            brightness_contrast_prob=0.03,
            brightness_range=0.1,
            constrast_range=0.1,
            gaussian_noise_prob=0.08,
            gaussian_noise_range=20.,  # variance range
            gaussian_blur_prob=0.03,
            gaussian_blur_kernel_range=(3, 3),
            image_compression_prob=0.1,
        )
        params = defaults
        scale_range_centered = tuple(x - 1 for x in params['scale_range'])
        params['scale_range'] = scale_range_centered

        tv_pp += [
            # this should be equivalent to initial resize & pad in Donut prepare_input()
            ResizeKeepRatio(image_size, longest=1, interpolation=input_cfg.interpolation),
            RandomCropOrPad(image_size, fill=input_cfg.fill_color),
        ]

        alb_pp += [
            BitmapAlb(p=params['bitmap_prob']),
            alb.OneOf([
                    ErosionAlb(params['erosion_dilation_scale']),
                    DilationAlb(params['erosion_dilation_scale'])
                ],
                p=params['erosion_dilation_prob']
            ),
            alb.Affine(
                shear={
                    "x": params['shear_range_x'],
                    "y": params['shear_range_y']
                },
                cval=input_cfg.fill_color,
                p=params['shear_prob']
            ),
            alb.ShiftScaleRotate(
                shift_limit_x=params['shift_range_x'],
                shift_limit_y=params['shift_range_y'],
                scale_limit=params['scale_range'],
                rotate_limit=params['rotate_range'],
                border_mode=border_mode,
                interpolation=interpolation_mode,
                value=input_cfg.fill_color,
                p=params['shift_scale_rotate_prob'],
            ),
            alb.GridDistortion(
                distort_limit=params['grid_distort_range'],
                border_mode=border_mode,
                interpolation=interpolation_mode,
                value=input_cfg.fill_color,
                p=params['grid_distort_prob'],
            ),
            alb.Compose(
                [
                    alb.Affine(
                        translate_px=(0, 5),
                        always_apply=True,
                        cval=input_cfg.fill_color,
                    ),
                    alb.ElasticTransform(
                        p=1.0,
                        alpha=params['elastic_alpha'],
                        sigma=params['elastic_sigma'],
                        alpha_affine=12.,  # FIXME no common param, alpha_affine unique to alb
                        border_mode=border_mode,
                        value=input_cfg.fill_color,
                    ),
                ],
                p=params['elastic_prob'],
            ),
            alb.RandomBrightnessContrast(
                brightness_limit=params['brightness_range'],
                contrast_limit=params['constrast_range'],
                brightness_by_max=True,
                p=params['brightness_contrast_prob'],
            ),
            alb.ImageCompression(
                quality_lower=95,
                p=params['image_compression_prob'],
            ),
            alb.GaussNoise(
                var_limit=params['gaussian_noise_range'],
                p=params['gaussian_noise_prob']
            ),
            alb.GaussianBlur(
                blur_limit=params['gaussian_blur_kernel_range'],
                p=params['gaussian_blur_prob'],
            ),
        ]
    else:
        # inference / eval
        tv_pp += [
            ResizeKeepRatio(image_size, longest=1, interpolation=input_cfg.interpolation),
            CenterCropOrPad(image_size, fill=input_cfg.fill_color),
        ]

    #alb_pp += [alb.pytorch.ToTensorV2()]
    if alb_pp:
        # FIXME leave alb uncomposed too if composed=False?
        tv_pp += [AlbWrapper(alb.Compose(alb_pp))]

    tv_pp += [transforms.ToTensor()]
    if do_normalize:
        tv_pp += [transforms.Normalize(input_cfg.mean, input_cfg.std)]

    return transforms.Compose(tv_pp) if composed else tv_pp
