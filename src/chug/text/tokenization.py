from functools import partial
from typing import Callable, Optional, Union

import torch


def prompt_end_pos(tokens: torch.Tensor, prompt_end_token_id: int, empty_default: int = 0) -> int:
    end_pos = torch.nonzero(tokens == prompt_end_token_id)
    return end_pos[-1].item() if end_pos.numel() > 0 else empty_default


def text_input_to_target(
        text_input_ids: torch.Tensor,
        tokenizer: Optional[Callable] = None,
        prompt_end_token: Optional[Union[str, int]] = None,
        ignore_id: int = -100,
        pad_token_id: Optional[int] = None,
):
    target = text_input_ids.clone()

    if pad_token_id is None:
        assert tokenizer is not None, 'tokenizer must be specified if pad_token_id is not.'
        pad_token_id = tokenizer.pad_token_id

    # model doesn't need to predict pad token
    target[target == pad_token_id] = ignore_id

    if prompt_end_token is not None:
        if isinstance(prompt_end_token, str):
            assert tokenizer is not None, 'tokenizer must be specified if prompt_end_token_id required.'
            prompt_end_token_id = tokenizer.convert_tokens_to_ids(prompt_end_token)
        else:
            prompt_end_token_id = prompt_end_token  # already an int

        # model doesn't need to predict prompt (for VQA)
        end_pos = prompt_end_pos(target, prompt_end_token_id)
        target[:end_pos + 1] = ignore_id

    return target


def tokenize(
        text: str,
        tokenizer: Callable,
        max_length: int,
        ids_only: bool = True,
):
    output = tokenizer(
        text,
        add_special_tokens=False,
        return_tensors="pt",
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )
    if ids_only:
        return output.input_ids[0]
    return output


def prepare_text_input(
        text_input,
        tokenizer: Callable,
        max_length: int,
        task_start_token: Optional[str] = None,
        prompt_end_token: Optional[str] = None,
        add_eos_token: bool = True,
        ignore_id: int = -100,
        include_target: bool = True,
        return_dict: bool = True,
        input_key: str = "text_input",
        target_key: str = "text_target",
):
    """
    Simple data preprocessing for raw-text data.
    """
    if task_start_token:
        text_input = task_start_token + text_input

    if add_eos_token:
        text_input += tokenizer.eos_token

    text_input_ids = tokenize(text_input, tokenizer=tokenizer, max_length=max_length)

    if include_target:
        text_target_ids = text_input_to_target(text_input_ids, tokenizer, prompt_end_token, ignore_id)
        if return_dict:
            return {input_key: text_input_ids, target_key: text_target_ids}
        else:
            return text_input_ids, text_target_ids
    else:
        # FIXME calculate prompt end pos for validation use (target not needed)
        if return_dict:
            return {input_key: text_input_ids}
        else:
            return text_input_ids


def create_text_preprocessor(
        tokenizer: Union[str, Callable],
        max_length: int = 1024,
        task_start_token: Optional[str] = None,
        prompt_end_token: Optional[str] = None,
        ignore_id: int = -100,
        include_target: bool = True,
        return_dict: bool = True,
        input_key: str = "text_input",
        target_key: str = "text_target",
):
    if isinstance(tokenizer, str):
        import transformers
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer)

    # FIXME just binding prepare_text_input fn for now
    #  This is currently tailored to the Donut style image enc + text decoder use case.
    #  More functionality / variety required.

    preprocess_fn = partial(
        prepare_text_input,
        tokenizer=tokenizer,
        max_length=max_length,
        task_start_token=task_start_token,
        prompt_end_token=prompt_end_token,
        ignore_id=ignore_id,
        include_target=include_target,
        return_dict=return_dict,
        input_key=input_key,
        target_key=target_key,
    )
    return preprocess_fn