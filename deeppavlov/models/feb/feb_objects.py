import pymorphy2 as pym
from razdel import sentenize, tokenize
import re

import string
from question2wikidata.server_queries import queries
# from FictionEmpatBot.queries.question2wikidata.server_queries import queries
from deeppavlov.core.common.log import get_logger
import requests
from dateutil import parser as date_parse

log = get_logger(__name__)

def var_dump(msg, header=''): print(f'-----{header}-----\n{msg}\n----\n')


class FebStopBranch(object):
    STOP = 'FebStopBranch.STOP'
    CONTINUE = 'FebStopBranch.CONTINUE'
    pass

class FebError(object):
    ET = 'et_'  # error type
    ET_SYS = 'et_sys_'  # system error
    ET_LOG = 'et_log_'  # busyness logic error
    ET_INP_DATA = 'et_log_inpdata_'  # input data check error

    EC = 'ec_'  # error cause
    EC_DATA = EC + 'data_'
    EC_DATA_LACK = EC_DATA + 'lack_'  # insufficient data
    EC_DATA_NONE = EC_DATA_LACK + 'none_'  # no value at all
    EC_DATA_DISABIG = EC_DATA + 'disambiguation_'  # there are two or more variants
    EC_DATA_VAL = EC_DATA + 'val_'  # wrong data value
    EC_DATA_TYPE = EC_DATA + 'type_'  # wrong data type
    EC_EXCEPTION = EC + 'exception_'
    EC_WRONG_TYPE = EC + 'wr'

    @staticmethod
    def is_err_subtype(err_code_1, err_code_2):
        """
        Check err_code_1 is special case (more detailed, i.e. longer) of err_code_2
        """
        return err_code_2.find(err_code_1, 0, len(err_code_1)) == 0

    def __init__(self, error_type, component, cause_d, text=None):
        """

        :param error_type: ET constant
        :param component: component where error occur
        :param cause_d: dict of error causes and its context
        """
        super().__init__()
        self.type = error_type
        self.component_name = type(component).__name__
        self.cause_d = cause_d


    def __repr__(self):
        vals = ', '.join(f'{k}={repr(v)}' for k, v in self.__dict__.items())
        return f'{self.__class__.__name__}({vals})'


class FebObject(object):

    def __init__(self, **kwargs):
        super().__init__()
        self.type = None  # object type
        self.errors = []  # errors list

    def add_error(self, error):
        print(f'new error {error}')
        self.errors.append(error)

    def __repr__(self):
        vals = ', '.join(f'{k}={repr(v)}' for k, v in self.__dict__.items())
        return f'{self.__class__.__name__}({vals})'

    def has_errors(self):
        return len(self.errors) != 0

    @classmethod
    def recursive_json(cls, obj):
        if isinstance(obj, list) or isinstance(obj, tuple) or isinstance(obj, set):
            # print(obj, type(obj))
            return [FebObject.recursive_json(element) for element in obj]
        elif isinstance(obj, dict):
            # print(obj, type(obj))s
            props_to_cut = FebObject.IGNORE_IN_DUMP.get(obj.__class__.__name__, [])
            return {k: FebObject.recursive_json(v) for k, v in obj.items() if k not in props_to_cut}
        elif isinstance(obj, FebObject) or isinstance(obj, FebError):
            # print(obj, type(obj))
            props_to_cut = FebObject.IGNORE_IN_DUMP.get(obj.__class__.__name__, [])
            return {k: FebObject.recursive_json(v) for k, v in obj.__dict__.items() if k not in props_to_cut}
        else:
            return obj

    IGNORE_IN_DUMP = {
        'FebToken': ['source_text'],
        # 'FebUtterance': ['re_text'],
        'FebIntent': ['confidence']
    }


class FebToken(FebObject):
    # Token types:
    PUNCTUATION = 't_punktuation'
    NOT_PUNCTUATION = 't_text'
    WORD = 't_word'
    NUMBER = 't_number'
    OTHER = 't_other'

    # Token language types:
    WORD_LANGUAGE_RU = 't_word_lng_ru'
    WORD_LANGUAGE_OTHER = 't_word_lng_other'

    # Token tags:
    TAG_EOFS = 'tag_eofs'
    TAG_AUTHOR = 'tag_author'
    TAG_BOOK = 'tag_book'

    @staticmethod
    def sentenize(text):
        """
        Split text into sentences
        :param text:
        :return: list [(start, stop, text), ... ]
        """
        return list(sentenize(text))

    @staticmethod
    def tokenize(text, flat=True, tag_eofs=True):
        sent_l = FebToken.sentenize(text)
        sent_tok_ll = [(list(map(lambda t: FebToken(sent.start + t.start, sent.start + t.stop,
                                                    t.text, source_text=text),
                                 tokenize(sent.text)))) for sent in sent_l]
        if tag_eofs:
            for tok_l in sent_tok_ll:
                if len(tok_l) > 0:
                    tok_l[-1].tags.add(FebToken.TAG_EOFS)
        if flat:
            return [tok for tok_l in sent_tok_ll for tok in tok_l]
        else:
            return sent_tok_ll

    @staticmethod
    def stemmer(sentence):
        from pymystem3 import Mystem
        STEMMER = Mystem()
        pos_map = {
            'A': 'ADJ',
            'ADV': 'ADV',
            'ADVPRO': 'ADV',
            'ANUM': 'ADJ',
            'APRO': 'DET',
            'COM': 'ADJ',
            'CONJ': 'SCONJ',
            'INTJ': 'INTJ',
            'NONLEX': 'X',
            'NUM': 'NUM',
            'PART': 'PART',
            'PR': 'ADP',
            'S': 'NOUN',
            'SPRO': 'PRON',
            'UNKN': 'X',
            'V': 'VERB'
        }
        processed = STEMMER.analyze(sentence)
        tagged = []
        for w in processed:
            try:
                lemma = w["analysis"][0]["lex"].lower().strip()
                text = w['text']
                pos = w["analysis"][0]["gr"].split(',')[0]
                pos = pos.split('=')[0].strip()
                pos = pos_map.get(pos, 'X')
                start = sentence.index(text)
                stop = start + len(text)
                token = FebToken(0, 0, text, normal_form=lemma, pos=pos)
                tagged.append(token)
            except (KeyError, IndexError):
                continue
        return tagged

    def __init__(self, start, stop, text, **kwargs):
        """

        :param start:
        :param stop:
        :param text:
        :param kwargs: source_text, t_type, lang
        """
        super().__init__()

        self.start = start
        self.stop = stop
        self.text = text
        self.source_text = kwargs.get('ttype', None)  # source text string
        if self.text:
            self.set_t_type()
        self.lang = kwargs.get('lang', None)  # token language
        self.normal_form = kwargs.get('normal_form', None)
        self.pos = kwargs.get('pos', None)
        self.tags = set()  # tokens tags (example: markers of NER)

    # TODO: identify other types
    def set_t_type(self, only_punktuation=True):
        assert only_punktuation, 'Other options not implemented!'
        if self.text:
            if re.match('.*[\w\d_].*', self.text) is None:  # TODO: check regexp
                self.type = FebToken.PUNCTUATION
            else:
                self.type = FebToken.NOT_PUNCTUATION
        else:
            raise ValueError('text value is not set')

    def set_pos(self, pos):
        self.pos = pos

    def set_normal_form(self, normal_form):
        self.normal_form = normal_form

    def __repr__(self):
        vals = str(self)
        return f'{self.__class__.__name__}({vals})'

    def __str__(self):
        rs = f'({self.start}, {self.stop}, {self.text}'
        if self.type:
            rs += f', type={self.type}'
        if self.lang:
            rs += f', lang={self.lang}'
        if self.tags:
            rs += f', tags={self.tags}'
        if self.pos:
            rs += f', pos={self.pos}'
        if self.normal_form:
            rs += f', normal_form={self.normal_form}'
        rs += ')'
        return rs

    def __eq__(self, other):
        return self.text == other.text


class FebEntity(FebObject):
    # class attributes:
    # types:
    AUTHOR = 'author'
    BOOK = 'book'
    GEOX = 'place'
    DATE = 'date'
    CHAR = 'char'
    OTHERS = 'others'
    GROUPING_SPACE_REGEX = re.compile('([^\w_\-\']|[+])', re.U)
    PUNCTATION_REGEX = re.compile('([^\w]|[+])', re.U)
    # TYPE_REGEX = re.compile('^(\w+(_characters|_author|_inspired_by))|author_name|author$')
    MORPH = pym.MorphAnalyzer()

    def __init__(self, type, **kwargs):
        super().__init__(**kwargs)
        self.type = type
        self.tokens = kwargs.get('tokens', None)  # list of tokens
        self.id = kwargs.get('id', None)
        self.qid = kwargs.get('qid', None)  # id in Wikidata
        self.qname = kwargs.get('qname', None)  # name in Wikidata
        self.normal_form = kwargs.get('normal_form', None)
        self.text_from_base = kwargs.get('text_from_base', None)
        self.first = kwargs.get('first', None)
        self.middle = kwargs.get('middle', None)
        self.last = kwargs.get('last', None)
        self.rollback_normal_form_capitalization()
        self.info = kwargs.get('info', None)

    def to_text(self):
        return ' '.join(t.text for t in self.tokens)  # todo rollback "t.text" from t['text']

    def tokens_to_search_string(self):
        return self.to_text()

    def to_values_dict(self):
        res_dict = {
            'type': self.type,
            'text': self.to_text(),
            'normal_form': self.normal_form,
            'qid': self.qid,
            'text_from_base': self.text_from_base
        }
        return res_dict

    @property
    def nomn(self):
        return self.inflect_string('nomn')

    @property
    def gent(self):
        return self.inflect_string('gent')

    @property
    def datv(self):
        return self.inflect_string('datv')

    @property
    def accs(self):
        return self.inflect_string('accs')

    @property
    def ablt(self):
        return self.inflect_string('ablt')

    @property
    def loct(self):
        return self.inflect_string('loct')

    def inflect_string(self, case):
        output_string = ''
        for token in self.simple_word_tokenize(self.text_from_base if self.text_from_base else self.normal_form):
            if token in string.punctuation or token.isspace():
                output_string = output_string + token  # ['О', "'", 'Нил', ',', 'Юджин', 'dsasa-dasdsa']
                continue

            correct_parse = self.choose_correct_parse(self.MORPH.parse(token))

            cased_tag = correct_parse.inflect({case})
            if not cased_tag:
                output_string = output_string + token
                continue

            cased_word = cased_tag.word
            output_string = output_string + cased_word
        output_string = self.rollback_string_capitalization(self.text_from_base if self.text_from_base else self.normal_form, output_string)
        return output_string

    def simple_word_tokenize(self, str):
        return [t for t in self.GROUPING_SPACE_REGEX.split(str)]

    @staticmethod
    def rollback_capitaliztion_from_text(str, new_string):
        text_tokens_list = re.sub("[^\w_]", " ", str).split()
        new_string_tokens_list = re.sub("[^\w_]", " ", new_string).split()
        for index, token in enumerate(text_tokens_list):
            if token[0].isupper():
                normal_form_right_cap = new_string_tokens_list[index][:1].upper() + new_string_tokens_list[index][1:]
                new_string_tokens_list[index] = normal_form_right_cap
                continue
        return ' '.join(new_string_tokens_list)

    def rollback_string_capitalization(self, str, new_string):
        string_tokenized = [t for t in self.PUNCTATION_REGEX.split(str)]
        new_string_tokenized = [t for t in self.PUNCTATION_REGEX.split(new_string)]
        correct_capitalized_string = ''
        if len(string_tokenized) != len(new_string_tokenized):  # нормальная форма не совпадает по пунктуации с текст
            return self.rollback_capitaliztion_from_text(str, new_string)
        else:
            for index, token in enumerate(string_tokenized):
                if token in string.punctuation or token.isspace():
                    correct_capitalized_string = correct_capitalized_string + token
                    continue
                if token[0].isupper():
                    correct_capitalized_string = correct_capitalized_string + new_string_tokenized[index][:1].upper() + \
                                           new_string_tokenized[index][1:]
                    continue
                else:
                    correct_capitalized_string = correct_capitalized_string + new_string_tokenized[index]
        return correct_capitalized_string

    def rollback_normal_form_capitalization(self):
        self.normal_form = self.rollback_string_capitalization(self.to_text(),self.normal_form) \
            if self.normal_form and self.tokens else self.normal_form

    def choose_correct_parse(self, parsed):
        correct_parse = None
        if isinstance(self, FebAuthor) or isinstance(self, FebChar):  # по типу вопрос
            for inx, parsed_word in enumerate(parsed):
                grams = parsed_word.tag
                if {'sing'} in grams:  # единственное число
                    if {'Name'} in grams or {'Surn'} in grams or {'Patr'} in grams or {'UNKN'} in grams:
                        correct_parse = parsed[inx]
                        break
        elif isinstance(self,FebBook):  # todo elif isinstance(self, FebGeox)
            correct_parse = parsed[0]
        else:
            for inx, found_word in enumerate(parsed):
                grams = found_word.tag
                if not {'Surn'} in grams and not {'Name'} in grams and not {'Patr'} in grams:
                    correct_parse = parsed[inx]
                    break
        if correct_parse is not None:
            # log.debug(f'---MorphAnalyzer___\n\n{correct_parse}\n')
            return correct_parse
        else:
            # log.debug(f'---MorphAnalyzer___\n\n{parsed[0]}\n')
            return parsed[0]


class FebAuthor(FebEntity):

    def __init__(self, **kwargs):
        super().__init__(FebEntity.AUTHOR, **kwargs)
        self.first = kwargs.get('first', None)
        self.middle = kwargs.get('middle', None)
        self.last = kwargs.get('last', None)

class FebBook(FebEntity):

    def __init__(self, **kwargs):
        super().__init__(FebEntity.BOOK, **kwargs)


class FebGeox(FebEntity):

    def __init__(self, **kwargs):
        super().__init__(FebEntity.GEOX, **kwargs)


class FebDate(FebEntity):

    def __init__(self, **kwargs):
        super().__init__(FebEntity.DATE, **kwargs)
        self.year()

    def year(self):
        if self.text_from_base:
            str_year = str(date_parse.parse(str(self.text_from_base)).year)
            self.text_from_base = str_year


class FebChar(FebEntity):

    def __init__(self, **kwargs):
        super().__init__(FebEntity.CHAR, **kwargs)


class FebOthers(FebEntity):

    def __init__(self, **kwargs):
        super().__init__(FebEntity.OTHERS, **kwargs)


class FebIntent(FebObject):
    """
    'book_author'
'book_written'
'book_published'
'book_characters'
'book_genre'
'book_main_theme'

'author_birthplace'
'author_productions'
'author_genres'
'author_when_born'
'author_where_lived'
'author_languages'
'author_when_died'
'author_where_died'
'author_where_buriedF'
'author_inspired_by'

    """
    supported_types = {q for q in queries.keys() if q[:5] != 'help_'}
    UNSUPPORTED_TYPE = 'unsupported_type'
    INTENT_NOT_SET_TYPE = 'intent_not_set_type'

    @classmethod
    def in_supported_types(cls, type):
        return type in cls.supported_types

    def __init__(self, type, **kwargs):
        super().__init__(**kwargs)

        self.type = type
        self.confidence = kwargs.get('confidence', 0.0)  # float confidence level

        self.result_qid = kwargs.get('result_qid', None)  # result id in Wikidata
        self.result_val = kwargs.get('result_val', None)  # result dict todo was 'result_str'

        # self.result_str = kwargs.get('result_str', None) # result string type

    @property
    def result_str(self):
        """
        Deprecated
        :return:
        """
        return str(self.result_val)

    # @property
    # def results_val_list_parse(self):
    #     results_keys_set = set(list([result.keys() for result in self.result_val][0]))
    #     parsed_result_dict = {}
    #     for result_key in results_keys_set:
    #         values_with_same_key_list = []
    #         for result in self.result_val:
    #             try:
    #                 value = result[result_key]
    #             except KeyError:
    #                 continue
    #             else:
    #                 if value:
    #                     values_with_same_key_list.append(value)
    #                 else:
    #                     continue
    #         if len(values_with_same_key_list) > 0:
    #             parsed_result_dict[result_key] = list(set(values_with_same_key_list))
    #
    #     return parsed_result_dict, list(results_keys_set)

    # @property
    # def results_val_list_parse(self):
    #     results_keys_set = set(list([result.keys() for result in self.result_val][0]))
    #     parsed_result_dict = {}
    #     for result_key in results_keys_set:
    #         values_with_same_key_list = []
    #         for result in self.result_val:
    #             try:
    #                 value = result[result_key]
    #             except KeyError:
    #                 continue
    #             else:
    #                 values_with_same_key_list.append(value)
    #
    #         parsed_result_dict[result_key] = list(set(values_with_same_key_list))
    #     log.info(f"__FebIntent__-results_val_list_parse___ parsed_result_dict : {parsed_result_dict}, result_keys: {results_keys_set}")
    #     return parsed_result_dict, list(results_keys_set)
    #
    # @property
    # def results_to_entities(self):
    #     results_dict,results_keys = self.results_val_list_parse
    #     for results_key in results_keys:
    #         results_val_list = results_dict[results_key]
    #         if results_key in ('authorLabel','authorlabel') :
    #             results_dict[results_key] = [FebAuthor(text_from_base=result_text) for result_text in results_val_list]
    #         elif results_key in ('bookLabel','booklabel'):
    #             results_dict[results_key] = [FebBook(text_from_base=result_text) for result_text in results_val_list]
    #         elif results_key in ('placeLabel','placelabel'):
    #             results_dict[results_key] = [FebGeox(text_from_base=result_text) for result_text in results_val_list]
    #         elif results_key == 'years':
    #             results_dict[results_key] = [FebDate(text_from_base=result_text) for result_text in results_val_list]
    #         elif results_key == ('charsLabel','charslabel'):
    #             results_dict[results_key] = [FebChar(text_from_base=result_text) for result_text in results_val_list]
    #         elif results_key in ('langLabel', 'genreLabel','subjLabel'):
    #             results_dict[results_key] = [FebOthers(text_from_base=result_text) for result_text in results_val_list]
    #         else:
    #             results_dict[results_key] = [FebEntity(type=None, text_from_base=result_text) for result_text in results_val_list]
    #     log.info(
    #         f"__FebIntent__-results_to_entities__ parsed_result_dict : {results_dict}, result_keys: {results_keys}")
    #     return results_dict, results_keys




class FebUtterance(FebObject):
    ERROR_IN_RESULT = 'error_in_result'

    def __init__(self, text, **kwargs):
        super().__init__(**kwargs)

        self.text = text  # input text
        self.get_suggested_text_from_yandex_speller()
        self.tokens = None  # list of tokens
        self.entities = []  # list of entities
        self.intents = []  # list of intents
        self.re_text = None  # responce text
        self.chat_id = kwargs.get('chat_id', None)  #ID отправителя
        self.pattern_type = kwargs.get('pattern_type', None)  #название шаблона в файле

    def to_dump(self):
        # return {k: [FebObject.recursive_json(item) for item in v if isinstance(item, (FebObject, FebError)) ] for k, v in self.__dict__.items() if v is not None }
        return {k: FebObject.recursive_json(v) for k, v in self.__dict__.items() if v is not None}

    def return_text(self):
        if self.re_text:
            return self.re_text
        else:
            return 'Что-то пошло не так, попробуйте еще раз.'

    def get_suggested_text_from_yandex_speller(self):
        response = self.request_spell_check()
        text_copy = self.text
        if response and len(response.json()) != 0:
            errors_list = response.json()
            for error in errors_list:
                if error['s'] is not None or len(error['s']) != 0:
                    self.text = self.text.replace(error['word'], error['s'][0])
            log.debug(f'---yandexspeller___\n\nErrors found, right text:   {self.text}\n \t Old text: {text_copy}')
        else:
            log.debug(f'---yandexspeller___\n\nNo errors found or yandex_speller failure:   {self.text}\n')
            self.text = self.text

    def request_spell_check(self):
        response = requests.get('https://speller.yandex.net/services/spellservice.json/checkText',
                                params={'text': self.text})
        if response.status_code == 200:
            if response is None:
                return None
            else:
                return response
        else:
            return None

    def get_gen_context(self) -> dict:
        gen_context = {}
        gen_context['params'] = [e for e in self.entities]  # [e.to_values_dict() for e in self.entities]

        if len(self.intents) > 0:
            # TODO: case of many intents
            intent = self.intents[0]
            gen_context['query_name'] = intent.type
            gen_context['log'] = log
            if intent.result_val:
                results = (intent.result_val, list(intent.result_val.keys()))  # results = intent.results_to_entities
                gen_context['results'] = results[0]
                gen_context['results_keys'] = results[1]
            else:
                gen_context['results'] = {'error': FebUtterance.ERROR_IN_RESULT}
                gen_context['results_keys'] = ['error']
        else:
            gen_context['query_name'] = FebIntent.INTENT_NOT_SET_TYPE
            gen_context['results'] = {'error': FebUtterance.ERROR_IN_RESULT}
            gen_context['results_keys'] = ['error']

        return gen_context
