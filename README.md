# Chugging Data 

A library to help w/ efficient training with multi-modal data. Initially focused on image & document + text tasks.

`chug` currently leverages `webdataset` and Huggingface `datasets`. `webdataset` tar files and dataset pipelines are preferred for scalable pretraining. For ease of use, Huggingface `datasets` are also supported and work great for exploration, validation, and fine-tune use cases.

## TODOs

### Nearish
* Cleanup and refinement, codebase will change.
* Documentation & unit-tests.
* Support reading of info .json/.yaml files for automatic shard info resolution for webdatasets (like timm).
* Support unified preprocessor functions for combined image + text tokenization (img+text token interleaving, etc.).
 
### Longish 
* Increase range of task pipelines for other tasks, modelling needs.
* Support additional modalities & targets (video, audio, detection/dense pixel targets, image/video/audio targets).
* Explore alternatives to .tar shards (array_record, arrow, etc).

## Usage / Examples

### Document Reading, Training w/ IDL
```python
import chug
img_cfg = chug.ImageInputCfg(size=(1024, 768), transform_type='doc_better')
img_fn = chug.create_image_preprocessor(input_cfg=img_cfg, is_training=True)
txt_fn = chug.create_text_preprocessor(
    'naver-clova-ix/donut-base',
    prompt_end_token='<s_idl>',
    task_start_token='<s_idl>',  # NOTE needs to be added to tokenizer
)

task_cfg = chug.DataTaskDocReadCfg(
    image_process_fn=img_fn,
    text_process_fn=txt_fn,
    page_sampling='random',
    error_handler='dump_and_reraise',
)
task_pipe = chug.create_task_pipeline(task_cfg)
data_cfg = chug.DataCfg(
    source='pipe:curl -s -f -L https://huggingface.co/datasets/pixparse/IDL-wds/resolve/main/idl-train-0{0000..1000}.tar',  # FIXME range
    split='train',
    batch_size=8,
    num_samples=1000000,  # FIXME get actual value
    num_workers=0,
    format='wds',
)
lb = chug.create_loader(
    data_cfg,
    task_cfg,
    is_training=True,
)
ii = iter(lb.loader)
sample = next(ii)
```

### Document Reading, Exploring IDL
```python
import chug
task_cfg = chug.DataTaskDocReadCfg(page_sampling='all')
task_pipe = chug.create_task_pipeline(task_cfg)

data_cfg = chug.DataCfg(
    source='pixparse/IDL-wds',
    split='train',
    batch_size=None,
    format='hfids',
    num_workers=0,    
)
lb = chug.create_loader(
    data_cfg,
    task_cfg,
)
ii = iter(lb.loader)
sample = next(ii)
```

### Document Reading, Training with PDFA

```python
import chug
img_cfg = chug.ImageInputCfg(size=(1024, 768), transform_type='doc_nougat')
img_fn = chug.create_image_preprocessor(input_cfg=img_cfg, is_training=True)
txt_fn = chug.create_text_preprocessor(
    'naver-clova-ix/donut-base',
    prompt_end_token='<s_pdfa>',
    task_start_token='<s_pdfa>',  # NOTE needs to be added to tokenizer
)

task_cfg = chug.DataTaskDocReadCfg(
    image_process_fn=img_fn,
    text_process_fn=txt_fn,
    page_sampling='random',
)
task_pipe = chug.create_task_pipeline(task_cfg)
data_cfg = chug.DataCfg(
    source='pipe:curl -s -f -L https://huggingface.co/datasets/pixparse/pdfa-english-train/resolve/main/pdfa-eng-train-{000000..005000}.tar',
    split='train',
    batch_size=8,
    num_samples=1000000,  # FIXME approx
    format='wds',   
)
lb = chug.create_loader(
    data_cfg,
    task_cfg,
    is_training=True,
)
ii = iter(lb.loader)
sample = next(ii)
```

### Document Reading, Exploring PDFA

```python
import chug

task_cfg = chug.DataTaskDocReadCfg(
    page_sampling='all',
)
task_pipe = chug.create_task_pipeline(task_cfg)
data_cfg = chug.DataCfg(
    source='pixparse/pdfa-eng-wds',
    split='train',
    batch_size=None,
    format='hfids',
    num_workers=0,
)
lb = chug.create_loader(
    data_cfg,
    task_cfg,
)
ii = iter(lb.loader)
sample = next(ii)
```


### Image + Text

### Training

```python
import chug
import transformers
from functools import partial
img_cfg = chug.ImageInputCfg(size=(512, 512), transform_type='image_timm')
img_fn = chug.create_image_preprocessor(input_cfg=img_cfg, is_training=True)
tokenizer = transformers.AutoTokenizer.from_pretrained('laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
txt_fn = partial(chug.tokenize, max_length=1000, tokenizer=tokenizer)
task_cfg = chug.DataTaskImageTextCfg(
    image_process_fn=img_fn,
    text_process_fn=txt_fn,
)
task_pipe = chug.create_task_pipeline(task_cfg)
data_cfg = chug.DataCfg(
    source='pipe:curl -s -f -L https://huggingface.co/datasets/pixparse/cc12m-wds/resolve/main/cc12m-train-{0000..2175}.tar',
    split='train',
    batch_size=8,
    num_samples=10000000,
    format='wds',   
)
lb = chug.create_loader(
    data_cfg,
    task_cfg,
    is_training=True,
)
ii = iter(lb.loader)
sample = next(ii)
```

### Document VQA

#### Training, Fine-tuning
```python
import chug
from chug.task_pipeline import create_task_pipeline
img_cfg = chug.ImageInputCfg(size=(1024, 768), transform_type='doc_basic')
img_fn = chug.create_image_preprocessor(img_cfg, is_training=True)
txt_fn = chug.create_text_preprocessor(
    'naver-clova-ix/donut-base-finetuned-docvqa',
    prompt_end_token='<s_answer>',
    task_start_token='<s_docvqa>',
)

task_cfg = chug.DataTaskDocVqaCfg(
    image_process_fn=img_fn,
    text_process_fn=txt_fn,
)
task_pipe = create_task_pipeline(task_cfg)

data_cfg = chug.DataCfg(
    source='pipe:curl -s -f -L https://huggingface.co/datasets/pixparse/docvqa-wds/resolve/main/docvqa-train-{000..383}.tar',
    split='train',
    batch_size=8,
    format='wds',
    num_samples=39463,
)
lb = chug.create_loader(
    data_cfg,
    task_cfg,
    is_training=True,
)
ii = iter(lb.loader)
sample = next(ii)
```

#### Exploration

```python
import chug
from chug.task_pipeline import create_task_pipeline
task_cfg = chug.DataTaskDocVqaCfg(
    question_prefix='Question: ',
    question_suffix='',
    answer_prefix='Answer: ',
    answer_suffix=''
)
task_pipe = create_task_pipeline(task_cfg)
data_cfg = chug.DataCfg(
    source='pixparse/docvqa-single-page-questions',
    split='validation',
    batch_size=None,
    format='hfids',
    num_workers=0,
)
lb = chug.create_loader(
    data_cfg,
    task_cfg
)
ii = iter(lb.loader)
```