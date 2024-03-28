from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


@dataclass
class DataTaskCfg:
    """
    Attributes:
        ...
        output_tuple: Output tuples instead of dicts from task pipeline.
        filter_valid: Filter out invalid / incomplete samples.
        flatten_json: Flatten json dicts into parent sample dict. Common to have in wds datasets.
        error_handler: 'specifies which error (exception) handler should be used, 'ignore_and_continue'
            is good for training, 'reraise_exception' for debugging purposes.
    """
    decode_fn: Optional[Callable] = None
    image_process_fn: Optional[Callable] = None
    text_process_fn: Optional[Callable] = None
    decode_and_process_fn: Optional[Callable] = None
    output_tuple: bool = False  # output features as tuple instead of dictionary
    filter_valid: bool = False  # enable filter to keep samples with valid key-values
    flatten_json: bool = True  # flatten nested 'json' dicts into parent sample
    error_handler: str = 'reraise_exception'