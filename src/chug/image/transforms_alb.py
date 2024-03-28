
try:
    import albumentations as alb
    from albumentations.pytorch import ToTensorV2
    has_albumentations = True
except ImportError:
    has_albumentations = False

try:
    import cv2
    has_cv2 = True
except ImportError:
    has_cv2 = False

import numpy as np


class AlbWrapper:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, im):
        return self.transforms(image=np.asarray(im))["image"]

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"        {t}"
        format_string += "\n)"
        return format_string


if has_albumentations and has_cv2:

    class ErosionAlb(alb.ImageOnlyTransform):
        def __init__(self, scale, always_apply=False, p=0.5):
            super().__init__(always_apply=always_apply, p=p)
            if type(scale) is tuple or type(scale) is list:
                assert len(scale) == 2
                self.scale = scale
            else:
                self.scale = (scale, scale)

        def get_transform_init_args_names(self):
            return ()

        def apply(self, img, **params):
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, tuple(np.random.randint(self.scale[0], self.scale[1] + 1, 2))
            )
            img = cv2.erode(img, kernel, iterations=1)
            return img


    class DilationAlb(alb.ImageOnlyTransform):
        def __init__(self, scale, always_apply=False, p=0.5):
            super().__init__(always_apply=always_apply, p=p)
            if type(scale) is tuple or type(scale) is list:
                assert len(scale) == 2
                self.scale = scale
            else:
                self.scale = (scale, scale)

        def get_transform_init_args_names(self):
            return ()

        def apply(self, img, **params):
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                tuple(np.random.randint(self.scale[0], self.scale[1] + 1, 2))
            )
            img = cv2.dilate(img, kernel, iterations=1)
            return img


    class BitmapAlb(alb.ImageOnlyTransform):
        def __init__(self, value=0, lower=200, always_apply=False, p=0.5):
            super().__init__(always_apply=always_apply, p=p)
            self.lower = lower
            self.value = value

        def get_transform_init_args_names(self):
            return ()

        def apply(self, img, **params):
            img = img.copy()
            img[img < self.lower] = self.value
            return img


    class CropMarginCv2:

        def __init__(self):
            pass

        def __call__(self, img):
            data = np.array(img.convert("L"))
            data = data.astype(np.uint8)
            max_val = data.max()
            min_val = data.min()
            if max_val == min_val:
                return img
            data = (data - min_val) / (max_val - min_val) * 255
            gray = 255 * (data < 200).astype(np.uint8)

            coords = cv2.findNonZero(gray)  # Find all non-zero points (text)
            a, b, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
            return img.crop((a, b, w + a, h + b))

