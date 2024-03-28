import io
import logging
import os
import random
import re
import warnings
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import webdataset as wds
from PIL import Image

from .helpers import log_and_continue


# IMPORTANT fitz aka PyMuPDF is AGPL licensed w/ a commercial purchase option, manual intervention required to use it.
_USE_AGPL_PYMUPDF = int(os.environ.get('CHUG_USE_AGPL_PYMUPDF', -1))
if _USE_AGPL_PYMUPDF < 1:
    import importlib
    if importlib.util.find_spec('fitz') is not None and _USE_AGPL_PYMUPDF == -1:
        # warn if not explicitly disable by setting to 0
        warnings.warn(
            "The fitz/pymupdf library is installed but disabled as the environment variable CHUG_USE_AGPL_PYMUPDF is"
            " not set to 1. Please be aware of the licensing concerns in your use cases if you enable it.")
    fitz = None
else:
    try:
        import fitz
    except ImportError as e:
        fitz = None

# defaults to pypdfium2 when fitz is not installed and/or not enabled
if fitz is None:
    try:
        import pypdfium2
    except ImportError as e:
        pypdfium2 = None
else:
    pypdfium2 = None

_logger = logging.getLogger(__name__)


PDF_EXTENSIONS = {
    'pdf',
}


def decode_pdf_pages(
        data: bytes,
        image_mode: str = 'L',
        page_indices: Optional[Tuple[int]] = None,
        select_random: Optional[int] = None,
        render_dpi: int = 144,
):
    rendered_pages = []

    with io.BytesIO(data) as b:
        # FIXME test and use an alternate pdf reader/render as default
        if fitz is not None:
            doc = fitz.Document(stream=b)
            num_doc_pages = doc.page_count

            if page_indices is not None:
                page_indices = [p % num_doc_pages for p in page_indices]  # support -ve indexing
            else:
                page_indices = range(num_doc_pages)

            if select_random:
                if select_random == 1:
                    page_indices = [random.choice(page_indices)]
                else:
                    page_indices = random.sample(page_indices, select_random)
                    page_indices.sort()

            for i in page_indices:
                page = doc.load_page(i)
                if image_mode == 'L':
                    fitz_cs = fitz.csGRAY
                    fitz_mode = 'L'
                    alpha = False
                elif image_mode == 'RGB' or image_mode == 'BGR':
                    fitz_cs = fitz.csRGB
                    fitz_mode = 'RGB'
                    alpha = False
                elif image_mode == 'RGBA' or image_mode == 'BGRA':
                    fitz_cs = fitz.csRGB
                    fitz_mode = 'RGBA'
                    alpha = True
                else:
                    assert False
                pixmap = page.get_pixmap(dpi=render_dpi, colorspace=fitz_cs, alpha=alpha)
                page_image = Image.frombuffer(fitz_mode, (pixmap.width, pixmap.height), pixmap.samples)
                if fitz_mode != page_image.mode:
                    page_image = page_image.convert(image_mode)

                rendered_pages += [page_image]

        elif pypdfium2 is not None:
            grayscale = image_mode == "L"
            reverse = 'RGB' in image_mode
            doc = pypdfium2.PdfDocument(data)
            num_doc_pages = len(doc)
            page_indices = page_indices or range(num_doc_pages)
            if select_random:
                page_indices = [random.choice(page_indices)]
            for i in page_indices:
                page = doc[i]
                page_image = page.render(
                    scale=render_dpi / 72,
                    grayscale=grayscale,
                    rev_byteorder=reverse,  # RGB instead of BGR(X)
                ).to_pil()
                if image_mode != page_image.mode:
                    page_image = page_image.convert(image_mode)

                rendered_pages += [page_image]
        else:
            assert False, "No PDF decoding library installed, please install one of pypdfium2 or fitz (PyMuPDF). " \
                          "NOTE: pypdifum2 is Apache 2.0 / BSD 3.0 licensed and fitz is AGPL."

        return rendered_pages, num_doc_pages


def decode_image_pages(
        data: bytes,
        image_mode: str = 'L',
        page_indices: Optional[Tuple[int]] = None,
        select_random: Optional[int] = None,
):
    """ decode multi-page image (e.g. TIFF)"""
    decoded_pages = []

    if isinstance(data, Image.Image):
        doc_image = data
    else:
        doc_image = Image.open(io.BytesIO(data))

    num_image_pages = getattr(doc_image, 'n_frames', 1)

    if page_indices is not None:
        page_indices = [p % num_image_pages for p in page_indices]  # support -ve indexing
    else:
        page_indices = range(num_image_pages)

    if select_random:
        if select_random == 1:
            page_indices = [random.choice(page_indices)]
        else:
            page_indices = random.sample(page_indices, select_random)
            page_indices.sort()

    for i, page_index in enumerate(page_indices):
        assert page_index < num_image_pages
        if num_image_pages > 1:
            doc_image.seek(page_index)
        else:
            assert page_index == 0, "not a multi-page image"
            doc_image.load()

        page_image = doc_image.convert(image_mode)
        decoded_pages.append(page_image)

    return decoded_pages, num_image_pages


class DecodeDoc:

    def __init__(
            self,
            imagespec,
            num_pages=1,
            page_sampling='first',
    ):
        """Create a PDF handler.

        Args:
            imagespec: short string indicating the type of decoding
                The `imagespec` specifies whether the image is decoded
                to numpy/torch/pi, decoded to uint8/float, and decoded
                to l/rgb/rgba:

                - l8: numpy uint8 l
                - rgb8: numpy uint8 rgb
                - rgba8: numpy uint8 rgba
                - l: numpy float l
                - rgb: numpy float rgb
                - rgba: numpy float rgba
                - torchl8: torch uint8 l
                - torchrgb8: torch uint8 rgb
                - torchrgba8: torch uint8 rgba
                - torchl: torch float l
                - torchrgb: torch float rgb
                - torch: torch float rgb
                - torchrgba: torch float rgba
                - pill: pil None l
                - pil: pil None rgb
                - pilrgb: pil None rgb
                - pilrgba: pil None rgba

        """
        if imagespec not in list(wds.autodecode.imagespecs.keys()):
            raise ValueError("Unknown imagespec: %s" % imagespec)
        self.imagespec = imagespec.lower()
        # FIXME need to work out padding / selection issues for multi-page support
        assert num_pages == 1, "Only 1-page decoding supported at present"
        self.num_pages = num_pages
        assert page_sampling in {'random', 'first', 'last'}  # TODO add 'all' w/ multi-page support
        self.page_sampling = page_sampling

    def __call__(self, key, data):
        """
        Args:
            key: file name extension
            data: data to be decoded
        """
        extension = re.sub(r".*[.]", "", key)
        if extension not in {'pdf', 'tiff', 'tif'}:
            print('skippy bippy')
            return None

        imagespec = self.imagespec
        atype, etype, mode = wds.autodecode.imagespecs[imagespec]

        select_random = False
        if self.page_sampling == 'random':
            page_indices = None
            select_random = True
        elif self.page_sampling == 'first':
            page_indices = [0]  # first page
        elif self.page_sampling == 'last':
            page_indices = [-1]
        else:
            assert False

        if extension == 'pdf':
            # pdf document
            result, num_pages = decode_pdf_pages(
                data,
                image_mode=mode.upper(),
                page_indices=page_indices,
                select_random=select_random,
            )
        else:
            # multi-page image doc (e.g. tiff)
            result, num_pages = decode_image_pages(
                data,
                image_mode=mode.upper(),
                page_indices=page_indices,
                select_random=select_random,
            )

        if atype == "pil":
            return result

        result = np.asarray(result)

        if etype == "float":
            result = result.astype(np.float32) / 255.0

        assert result.ndim in [2, 3], result.shape
        assert mode in ["l", "rgb", "rgba"], mode

        if mode == "l":
            if result.ndim == 3:
                result = np.mean(result[:, :, :3], axis=2)
        elif mode == "rgb":
            if result.ndim == 2:
                result = np.repeat(result[:, :, np.newaxis], 3, axis=2)
            elif result.shape[2] == 4:
                result = result[:, :, :3]
        elif mode == "rgba":
            if result.ndim == 2:
                result = np.repeat(result[:, :, np.newaxis], 4, axis=2)
                result[:, :, 3] = 255
            elif result.shape[2] == 3:
                result = np.concatenate(
                    [result, 255 * np.ones(result.shape[:2])], axis=2
                )

        assert atype in ["numpy", "torch"], atype

        if atype == "numpy":
            return result
        elif atype == "torch":
            import torch

            if result.ndim == 3:
                return torch.from_numpy(result.transpose(2, 0, 1))
            else:
                return torch.from_numpy(result)

        return None


def create_image_decoder(
        decode_fn: Optional[Callable] = None,
        image_mode: str = "RGB",
        enable_doc: bool = False,  # FIXME enable doc support by default once tested?
        handler: Callable = log_and_continue,
):
    if decode_fn is None:
        if image_mode == "L":
            img_type = "pill"
        elif image_mode == "RGB":
            img_type = "pilrgb"
        else:
            assert False, f"Unsupported image_mode ({image_mode})"
        if enable_doc:
            # FIXME, generic img + pdf decode WIP
            decode_fn = wds.decode(DecodeDoc(img_type), img_type, handler=handler)
        else:
            decode_fn = wds.decode(img_type, handler=handler)
    elif isinstance(decode_fn, (list, tuple)):
        decode_fn = wds.decode(*decode_fn, handler=handler)
    else:
        assert isinstance(decode_fn, Callable)
        decode_fn = wds.map(decode_fn)

    return decode_fn