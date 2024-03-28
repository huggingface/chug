import random
from typing import Callable, Dict, List, Optional, Tuple

from chug import FeatureInfo, ImageFeatureInfo
from chug.doc import DocProcessor, DEFAULT_QUESTION_FEAT, DEFAULT_QUESTION_ID_FEAT, DEFAULT_ANSWER_FEAT, \
    DEFAULT_DOC_FEAT
from chug.doc.doc_processor import _get_value


class DocVqaProcessor(DocProcessor):
    def __init__(
            self,
            image_process_fn: Optional[Callable] = None,
            text_process_fn: Optional[Callable] = None,
            question_feat: FeatureInfo = DEFAULT_QUESTION_FEAT,
            question_id_feat: FeatureInfo = DEFAULT_QUESTION_ID_FEAT,
            answer_feat: FeatureInfo = DEFAULT_ANSWER_FEAT,
            multi_qa_feat: Optional[FeatureInfo] = None,
            image_input_feat: ImageFeatureInfo = DEFAULT_DOC_FEAT,
            text_target_feat: FeatureInfo = FeatureInfo('text_target', input_key=None),
            question_prefix: Optional[str] = '<s_question>',
            question_suffix: Optional[str] = '</s_question>',
            answer_prefix: Optional[str] = '<s_answer>',
            answer_suffix: Optional[str] = '</s_answer>',
            render_dpi: int = 150,
            squeeze_pages: bool = True,
            expand_pages: bool = False,
            flatten_json: bool = True,
            seed: int = 0,
    ):
        super().__init__(
            image_process_fn=image_process_fn,
            text_process_fn=text_process_fn,
            image_input_feat=image_input_feat,
            text_target_feat=text_target_feat,
            render_dpi=render_dpi,
            squeeze_pages=squeeze_pages,
            expand_pages=expand_pages,
            flatten_json=flatten_json,
            seed=seed,
        )
        self.question_feat = question_feat
        self.question_key = question_feat.input_key.split(';')
        self.question_id_feat = question_id_feat
        self.question_id_key = question_id_feat.input_key.split(';')
        self.answer_feat = answer_feat
        self.answer_key = answer_feat.input_key.split(';')
        if multi_qa_feat is not None:
            self.expand_pages = True  # override
            self.multi_qa_key = multi_qa_feat.input_key.split(';')
        else:
            # expand pages only used / supported for multi-qa expansion right now
            self.expand_pages = False
            self.multi_qa_key = None

        # FIXME support flexible q/a prompting formats, do with prefix/suffix or template strings?
        # Donut style: '<s_docvqa><s_question>{question}</s_question><s_answer>{answer}</answer><eos>'
        # Common: 'Question: {question} Answer: {answer}<eos>'
        self.question_prefix = question_prefix or ''
        self.question_suffix = question_suffix or ''
        self.answer_prefix = answer_prefix or ''
        self.answer_suffix = answer_suffix or ''
        #self.prompt_template = '<s_question>{question}</s_question><s_answer>'
        #self.prompt_template_full = '<s_question>{question}</s_question><s_answer>{answer}</s_answer>'

    def _decode_anno(self, sample) -> Tuple[Dict, List[int], int]:
        if self.multi_qa_key:
            # FIXME multi qa expansion is a WIP
            qa_list = sample[self.multi_qa_key]
            assert isinstance(qa_list, (list, tuple)), f'Expected a list or tuple, got {type(qa_list)}.'
            assert False, 'WIP'
        else:
            question = _get_value(self.question_key, sample)
            question_id = _get_value(self.question_id_key, sample)
            answers = _get_value(self.answer_key, sample)

            if answers is not None and self.text_target_feat is not None:
                answer = random.choice(answers)
            else:
                answer = None

            input_text = self.question_prefix + question + self.question_suffix + self.answer_prefix
            if answer is not None:
                input_text += answer + self.answer_suffix

            output = {
                self.text_input_name: input_text,
                '_parse': {
                    'question_id': question_id,
                    'question': question,
                    'answers': answers,  # list, all answers included in parse
                }
            }

            if self.text_process_fn is not None:
                processed = self.text_process_fn(input_text)
                assert self.text_input_name in processed, \
                    f"Text input name '{self.text_input_name}' not found in processed sample."
                if self.text_target_feat is not None:
                    assert self.text_target_name in processed, f"Expected a text target named '{self.text_target_name}' in processed sample."
                output.update(processed)
            else:
                output[self.text_input_name] = input_text

            return output, [0], 1

    def _expand_anno(self, anno, count: int):
        # FIXME implement expansion for multi-qa (and eventually multi-page option)
        pass
