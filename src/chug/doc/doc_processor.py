import abc
import json
import logging
import random
from typing import Callable, Dict, List, Optional, Tuple

from chug.common import FeatureInfo, ImageFeatureInfo
from chug.wds import decode_image_pages, decode_pdf_pages


from .constants import (
    DEFAULT_DOC_FEAT,
)

_logger = logging.getLogger(__name__)


def get_next_valid_page_index(
        current_index: int,
        num_pages: int,
        page_annos: list,
        retries: int = 10,
        wanted: str = 'lines',
):
    """
    Get the index of the next valid page which contains text. If it doesn't find any non-empty page
    after 'retries' attempts, it raises a RuntimeError.

    Parameters:
    current_index (int): Current page index.
    num_pages (int): Total number of pages.
    page_annos (list): List of page annotations.
    retries (int): Number of maximum retries for a given document.

    Returns:
    int: The index of the next non-empty page.
    """
    for _ in range(retries):
        # Get the next index, wrap around to 0 if it exceeds num_pages (in case of random init)
        current_index = (current_index + 1) % num_pages
        anno_page = page_annos[current_index]
        anno_page = anno_page.get(wanted, anno_page)  # use 'lines' / 'words' level if exists
        if anno_page["text"]:
            return current_index
    raise RuntimeError(f"No non-empty page found after {retries} attempts")


def _get_value(keys, sample):
    if isinstance(keys, (list, tuple)):
        value = None
        for k in keys:
            if (value := sample.get(k, None)) is not None:
                break
        return value
    else:
        return sample.get(keys, None)


class DocProcessor:
    """ Process documents w/ OCR annotation for reading tasks.
    """

    def __init__(
            self,
            image_process_fn: Optional[Callable] = None,
            text_process_fn: Optional[Callable] = None,
            image_input_feat: ImageFeatureInfo = DEFAULT_DOC_FEAT,
            text_input_feat: FeatureInfo = FeatureInfo('text_input', input_key='pages'),
            text_target_feat: FeatureInfo = FeatureInfo('text_target', input_key=None),
            page_sampling: str = 'random',
            render_dpi: int = 150,
            squeeze_pages: bool = True,
            expand_pages: bool = False,
            flatten_json: bool = True,
            seed: int = 0,
    ):
        """

        Args:
            image_process_fn:
            text_process_fn:
            page_sampling:
            render_dpi:
            seed:
        """
        self.image_process_fn = image_process_fn
        self.text_process_fn = text_process_fn

        self.image_input_feat = image_input_feat
        self.image_input_name = image_input_feat.output_name
        self.image_input_key = image_input_feat.input_key.split(';')
        self.text_input_feat = text_input_feat
        self.text_input_name = text_input_feat.output_name
        self.text_input_key = text_input_feat.input_key.split(';')
        self.text_target_feat = text_target_feat
        self.text_target_name = text_target_feat.output_name

        self.page_sampling = page_sampling
        self.render_dpi = render_dpi
        self.squeeze_pages = squeeze_pages
        self.expand_pages = expand_pages
        self.flatten_json = flatten_json
        self.generator = random.Random()
        self.generator.seed(seed)
        # FIXME note, should move to torchvision v2 annotations at some point
        #  * they should all eventually have a generator arg for better handling random state
        #  * they have forms that accept bbox/points args to transform annotations in sync with image

    def _preprocess_image_pages(self, decoded_pages, page_image_info=None):
        if self.image_process_fn is None:
            return decoded_pages

        if page_image_info is not None:
            # FIXME, WIP. If train objective involves masking or otherwise processing image
            #  with knowledge of annotations / text content, anno info should contain
            #  mask locations, etc. For such a task, we need to pass it to image preprocess
            decoded_pages = [self.image_process_fn(dp, page_info=pi) for dp, pi in zip(decoded_pages, page_image_info)]
        else:
            decoded_pages = [self.image_process_fn(dp) for dp in decoded_pages]

        return decoded_pages

    def _decode_image_pages(
            self,
            sample,
            ext,
            page_indices,
            num_anno_pages,
    ):
        image_mode = self.image_input_feat.image_mode

        decoded_pages, num_image_pages = decode_image_pages(
            sample[ext],
            image_mode=image_mode,
            page_indices=page_indices,
        )
        if num_image_pages != num_anno_pages:
            _logger.warning(
                f'Mismatch between num image and num annotation pages {num_image_pages} != {num_anno_pages}'
                f' for sample {sample["__url__"]}, {sample["__key__"]}.')

        decoded_pages = self._preprocess_image_pages(decoded_pages)

        return decoded_pages, num_image_pages

    def _decode_pdf_pages(
            self,
            sample,
            ext,
            page_indices,
            num_anno_pages,
    ):
        image_mode = self.image_input_feat.image_mode
        decoded_pages, num_image_pages = decode_pdf_pages(
            sample[ext],
            image_mode=image_mode,
            page_indices=page_indices,
        )
        if num_anno_pages is not None and num_image_pages != num_anno_pages:
            _logger.warning(
                f'Mismatch between num image and num annotation pages {num_image_pages} != {num_anno_pages}'
                f' for sample {sample["__url__"]}, {sample["__key__"]}.')

        decoded_pages = self._preprocess_image_pages(decoded_pages)

        return decoded_pages, num_image_pages

    @abc.abstractmethod
    def _decode_anno(self, sample) -> Tuple[Dict, List[int], int]:
        pass

    def _expand_anno(self, anno, count: int):
        expanded_annos = [
            {k: v[i] if isinstance(v, (list, tuple)) else v for k, v in anno.items()}
            for i in range(count)
        ]
        return expanded_annos

    def _squeeze_anno(self, anno):
        anno = {k: v[0] if isinstance(v, (list, tuple)) else v for k, v in anno.items()}
        return anno

    def __call__(self, sample):
        if 'json' in sample and isinstance(sample['json'], bytes):
            # decode json if present and in undecoded state
            sample['json'] = json.loads(sample['json'])

        if self.flatten_json and 'json' in sample:
            # flatten json into sample
            sample.update(sample.pop('json'))

        # FIXME separate decode & preprocess interfaces

        # decode page annotations / text
        page_anno, page_indices, num_anno_pages = self._decode_anno(sample)

        # decode page images
        page_images = []
        for ext in self.image_input_key:
            if ext in sample:
                if ext == 'pdf':
                    images, num_image_pages = self._decode_pdf_pages(
                        sample,
                        ext,
                        page_indices,
                        num_anno_pages,
                    )
                else:
                    images, num_image_pages = self._decode_image_pages(
                        sample,
                        ext,
                        page_indices,
                        num_anno_pages,
                    )
                page_images.extend(images)
                # process one document type per doc, should be ordered by priority
                break

        assert len(page_images), 'No page images present'

        if self.expand_pages and len(page_images) > 1:
            # expand pages and page annotations into multiple samples (return list of sample dicts)
            page_anno = self._expand_anno(page_anno, len(page_images))
            decoded = [{self.image_input_name: pi, **pa} for pi, pa in zip(page_images, page_anno)]
        else:
            if self.squeeze_pages and len(page_images) == 1:
                # squeeze page & annotation lists into singular items
                page_images = page_images[0]
                page_anno = self._squeeze_anno(page_anno)
            decoded = {self.image_input_name: page_images, **page_anno}

        return decoded


