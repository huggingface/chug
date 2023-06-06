from typing import Callable

import webdataset as wds

from .loader import log_and_continue


def filter_no_caption_or_no_image(sample):
    has_caption = ('txt' in sample)  # FIXME configurable
    has_image = ('png' in sample or 'jpg' in sample or 'jpeg' in sample or 'webp' in sample)  # FIXME configurable
    return has_caption and has_image


def create_image_text_pipe(
    image_preprocess,
    text_tokenizer,
    decoder=None,
    as_tuple=True,
):
    if decoder is None or isinstance(decoder, str):
        img_type = decoder or "pilrgb"
        decoder = wds.decode(img_type, handler=log_and_continue)
    elif isinstance(decoder, (list, tuple)):
        decoder = wds.decode([decoder], handler=log_and_continue)
    else:
        assert isinstance(decoder, Callable)

    pipe = [
        wds.select(filter_no_caption_or_no_image),
        decoder,
        wds.rename(image="jpg;png;jpeg;webp", text="txt"),  # FIXME make mapping configurable
        wds.map_dict(image=image_preprocess, text=text_tokenizer),
    ]
    if as_tuple:
        pipe += [wds.to_tuple("image", "text")]
    return pipe
