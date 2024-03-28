from chug import FeatureInfo, ImageFeatureInfo

DEFAULT_DOC_KEY = "pdf;tif;png;jpeg;jpg;webp;image"
DEFAULT_QUESTION_KEY = "question;query"
DEFAULT_QUESTION_ID_KEY = "question_id;query_id"
DEFAULT_ANSWER_KEY = "answer;answers"

DEFAULT_DOC_KEY_TUPLE = tuple(DEFAULT_DOC_KEY.split(';'))
DEFAULT_QUESTION_KEY_TUPLE = tuple(DEFAULT_QUESTION_KEY.split(';'))
DEFAULT_ANSWER_KEY_TUPLE = tuple(DEFAULT_ANSWER_KEY.split(';'))

DEFAULT_DOC_FEAT = ImageFeatureInfo('image_input', input_key=DEFAULT_DOC_KEY, image_mode='L')
DEFAULT_QUESTION_FEAT = FeatureInfo(None, input_key=DEFAULT_QUESTION_KEY)
DEFAULT_QUESTION_ID_FEAT = FeatureInfo(None, input_key=DEFAULT_QUESTION_ID_KEY)
DEFAULT_ANSWER_FEAT = FeatureInfo(None, input_key=DEFAULT_ANSWER_KEY)
