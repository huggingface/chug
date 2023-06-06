import json
import io
import random
from functools import partial

import webdataset as wds

from PIL import Image

from .loader import log_and_continue


def filter_no_annotation_or_no_image(sample):
    # FIXME check sample for valid doc/image + annotation
    return True


def merge_lines(page_anno, tokenizer):
    output = '\n'.join([l['text'] for l in page_anno['lines']])
    output = tokenizer(output)
    return output


class DocProcessor:

    def __init__(
            self,
            image_preprocess,
            anno_preprocess,
            image_key='tif;tiff;png',
            image_format='L',
            seed=0,
    ):
        self.image_preprocess = image_preprocess
        self.anno_preprocess = anno_preprocess
        self.image_ext = image_key.split(';')
        self.image_fmt = image_format
        self.rng = random.Random()
        self.rng.seed(seed)

    def _page_text(self, anno, page_indices=None):
        page_indices = page_indices or range(len(anno['pages']))
        text_tokens = []
        for i in page_indices:
            page_tokens = self.anno_preprocess(anno['pages'][i])
            text_tokens.append(page_tokens)
        return text_tokens

    def __call__(self, sample):
        anno = json.loads(sample['json'])

        num_pages = len(anno['pages'])

        # FIXME for initial behaviour we will randomly sample one of N pages
        # TODO determine if we want to train in
        page_indices = [self.rng.randint(0, num_pages)]

        # decode image
        page_images = []
        page_text = []
        for ext in self.image_ext:
            if ext in sample:
                with io.BytesIO(sample[ext]) as b:
                    img = Image.open(b)

                assert img.n_frames == num_pages

                for i in page_indices:
                    page = img.seek(i)
                    if self.image_fmt:
                        page = page.convert(self.image_fmt)
                    page = self.image_preprocess(page)
                    page_images.append(page)

                page_text = self._page_text(anno, page_indices)

        if num_pages == 1:
            # FIXME always list?
            page_images = page_images[0]
            page_text = page_text[0]

        decoded = dict(image=page_images, text=page_text)
        return decoded


def _decode_samples(
        data,
        decoder,
        handler=log_and_continue,
):
    """Decode samples with skip."""
    for sample in data:
        try:
            result = decoder(sample)
        except Exception as exn:
            if handler(exn):
                continue
            else:
                break

        # empty results are skipped
        if result is not None:
            if isinstance(sample, dict) and isinstance(result, dict):
                result["__key__"] = sample.get("__key__")
            yield result


def create_doc_anno_pipe(
    image_preprocess,
    anno_preprocess,
    # page_sampling='',
):
    pipe = [
        wds.select(filter_no_annotation_or_no_image),
        # document decoding & pre-processing done together, there is coupling in random page
        # selection and possibly pre-processing / masking of image vs text
        partial(
            _decode_samples,
            decoder=DocProcessor(
                image_preprocess=image_preprocess,
                anno_preprocess=anno_preprocess,
            ),
        ),
        # need to think tuple conversion / downstream keys from anno preprocess
        wds.to_tuple("image", "text"),
    ]
    return pipe
