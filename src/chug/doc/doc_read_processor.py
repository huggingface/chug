from typing import Optional, Callable

from chug import ImageFeatureInfo, FeatureInfo
from chug.doc import DocProcessor, DEFAULT_DOC_FEAT
from chug.doc.doc_processor import get_next_valid_page_index, _get_value, _logger


class DocReadProcessor(DocProcessor):
    """ Process documents w/ OCR annotation for reading tasks.
    """

    def __init__(
            self,
            image_process_fn: Optional[Callable] = None,
            text_process_fn: Optional[Callable] = None,
            image_input_feat: ImageFeatureInfo = DEFAULT_DOC_FEAT,
            text_input_feat: FeatureInfo = FeatureInfo('text_input', input_key='pages'),
            text_target_feat: FeatureInfo = FeatureInfo('text_target', input_key=None),
            line_break: str = '\n',
            page_sampling: str = 'random',
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
            text_input_feat=text_input_feat,
            text_target_feat=text_target_feat,
            render_dpi=render_dpi,
            page_sampling=page_sampling,
            squeeze_pages=squeeze_pages,
            expand_pages=expand_pages,
            flatten_json=flatten_json,
            seed=seed,
        )
        self.line_break = line_break
        assert page_sampling in ('random', 'first', 'all_valid', 'all')

    def _process_anno_pages(self, anno):
        assert isinstance(anno, (list, tuple)), f"Annotation should be a list of pages"
        num_pages = len(anno)
        if not num_pages:
            raise RuntimeError("Empty annotation. Skipping...")

        # FIXME for initial behaviour we will randomly sample one of N pages
        # TODO determine if we want to train in multi-page mode, use another sampling strategy?
        page_indices = []
        try:
            if self.page_sampling == 'random':
                n_wanted_pages = min(1, num_pages)  # TODO increase for multi-page processing, rand start+end?
                current_index = self.generator.randrange(-1, num_pages - 1)
                for _ in range(n_wanted_pages):
                    current_index = get_next_valid_page_index(current_index, num_pages, anno)
                    page_indices.append(current_index)
            elif self.page_sampling == 'first':
                current_index = get_next_valid_page_index(-1, num_pages, anno)
                page_indices.append(current_index)
            elif self.page_sampling == 'all_valid':
                current_index = -1
                for _ in range(num_pages):
                    current_index = get_next_valid_page_index(current_index, num_pages, anno)
                    page_indices.append(current_index)
            elif self.page_sampling == 'all':
                page_indices = list(range(num_pages))
        except RuntimeError:
            pass

        if not page_indices:
            raise RuntimeError("No valid annotated pages. Skipping...")

        text_pages = []
        tokenized_text_pages = []
        target_pages = []
        for current_index in page_indices:
            # FIXME currently encoding each page separately with own start/end tokens.
            #  For multi-age should consider encoding in one sequence w/ page-break tokens.
            anno_page = anno[current_index]
            if 'lines' in anno_page:
                # Two supported formats right now
                # {
                #     'pages': [
                #         {
                #             'text': [],  # these are lines
                #             'bbox': [],
                #         }
                #     ]
                # }
                #
                # OR
                #
                # {
                #     'pages': [
                #         {
                #             'lines': {
                #                 'text': [],
                #                 'bbox': [],
                #             },
                #             'words': {
                #                 'text': [],
                #                 'bbox': [],
                #             }
                #         }
                #     ]
                # }
                #
                #
                anno_page = anno_page['lines']

            # Currently page text is created by concatenating lines of text with a CR line break
            # Additions could involve:
            # * using different line-break tokens between lines
            # * using word-level bbox anno information to mask works and construct partial lines
            # * group lines into blocks (or use block annos) and treat blocks / paragraphs of text and
            if not anno_page["text"]:
                raise RuntimeError("No text on page, skipping sample...")

            text = self.line_break.join(anno_page["text"])

            # FIXME cleanup, split process and decode for more flexibility
            # tokenize w/ and generate training target if enabled
            if self.text_process_fn is not None:
                processed = self.text_process_fn(text)
                assert self.text_input_name in processed, \
                    f"Text input name '{self.text_input_name}' not found in processed sample."
                tokenized_text_pages.append(processed[self.text_input_name])
                if self.text_target_name in processed:
                    target_pages.append(processed[self.text_target_name])
                else:
                    if self.text_target_feat is not None:
                        assert False, f"Expected a text target named '{self.text_target_name}' in processed sample."
            else:
                # FIXME warn assert that target not supported w/o text preprocessing?
                tokenized_text_pages.append(text)

            text_pages.append(anno_page["text"])  # unencoded text added as lines

        gt_parse = {
            'num_pages': num_pages,  # total # of pages in doc
            'page_indices': page_indices,  # page indices sampled
            'page_text': text_pages,  # text of sampled page indices pages[].lines[]
        }

        output = {
            self.text_input_name: tokenized_text_pages,
            '_parse': gt_parse,
        }
        if target_pages:
            output[self.text_target_name] = target_pages

        return output

    def _decode_anno(self, sample):
        anno = _get_value(self.text_input_key, sample)
        assert anno is not None, f"No annotation found with keys ({self.text_input_key})."

        try:
            page_anno = self._process_anno_pages(anno)
        except Exception as exn:
            _logger.error(f'Issue processing annotation for {sample["__url__"]}, {sample["__key__"]}.')
            #_logger.error(json.dumps(anno, indent=4))
            raise exn

        # extract info from the _parse
        info = page_anno.get('_parse', {})
        page_indices = info.get('page_indices', [0])  # the samples page indices
        num_anno_pages = info.get('num_pages', 1)

        # TODO support 'image info' to relay details such as text bbox, layout
        # page_image_info = info.get('image_info', None)
        # if page_image_info is not None:
        #     assert len(page_image_info) == len(page_indices)

        return page_anno, page_indices, num_anno_pages

    def _expand_anno(self, anno, count: int):
        expanded_annos = []
        for i in range(count):
            sample = {}
            for k, v in anno.items():
                if k == '_parse':
                    gt_parse = {}
                    gt_parse['num_pages'] = v['num_pages']
                    gt_parse['page_indices'] = [v['page_indices'][i]]
                    gt_parse['page_text'] = [v['page_text'][i]]
                    sample[k] = gt_parse
                else:
                    sample[k] = v[i] if isinstance(v, (list, tuple)) else v
            expanded_annos.append(sample)
        return expanded_annos
