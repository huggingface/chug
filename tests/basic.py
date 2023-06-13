

from timm.data import create_transform
from transformers import AutoTokenizer

from chug import create_image_text_pipe, create_wds_loader

mean = (0.5,) * 3
std = (0.5,) * 3

image_preprocess = create_transform(
    input_size=448,
    is_training=True,
    hflip=0.,
    mean=mean,
    std=std,
)

tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
text_preprocess = lambda x: tokenizer(
    x,
    return_tensors='pt',
    max_length=77,
    padding='max_length',
    truncation=True,
).input_ids

decoder_pipe = create_image_text_pipe(
    image_preprocess=image_preprocess,
    text_preprocess=text_preprocess,
)

ll = create_wds_loader(
    '/cc12m/cc12m-train-{0000..2175}.tar',
    decoder_pipe,
    is_train=True,
    num_samples=int(11e6),  # very approx
    batch_size=64,
)

for i, (img, text) in enumerate(ll.loader):
    print(i, img.shape, text.shape)
    if i > 100:
        break