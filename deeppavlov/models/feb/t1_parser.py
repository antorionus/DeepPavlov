# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from overrides import overrides

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.common.log import get_logger
import re

from .feb_objects import *
from .feb_common import FebComponent

log = get_logger(__name__)


@register('feb_t1_parser')
class FebT1Parser(FebComponent):
    """Convert batch of strings
    sl = ["author_birthplace author Лев Николаевич Толстой",
      -(to)->
        utterence object
      """
    @classmethod
    def component_type(cls):
        return cls.START_COMPONENT

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def test_and_prepare(self, utt):
        """
        Test input data and prepare data to process
        :param in_obj:
        :return: list (even if there is only one object to process!) of tuple(object, context)
            object - object for processing (must be instanceof FebObject)
            context - dictionary with context for processing
        """
        # if not isinstance(in_obj, str):
        #     raise TypeError(f"FebT1Parser is not implemented for `{type(in_obj)}`")
        # utt = FebUtterance(in_obj)
        utt.tokens = FebToken.tokenize(utt.text)
        tokens = [t for t in utt.tokens if t.type != FebToken.PUNCTUATION]
        if len(tokens) < 3:
            utt.add_error(FebError(FebError.ET_INP_DATA, self, {FebError.EC_DATA_LACK: tokens}))
        return [(utt, {'intents_code':tokens[0],
                           'entity_code': tokens[1],
                           'entity_text': tokens[2:]})]

    def process(self, obj, context):
        """
        Main processing function
        :param obj: obj to process
        :param context: dict with processing context
        :return: processed object
        """
        utt = obj
        intents_code = context['intents_code'].text
        entity_code = context['entity_code'].text
        entity_text = context['entity_text']
        if FebIntent.in_supported_types(intents_code):
            intent = FebIntent(intents_code)
        else:
            intent = FebIntent(FebIntent.UNSUPPORTED_TYPE)
            intent.add_error(FebError(FebError.ET_INP_DATA, self, {FebError.EC_DATA_VAL: intents_code}))
        utt.intents.append(intent)

        if entity_code == FebEntity.AUTHOR:
            utt.entities.append(FebAuthor(tokens=entity_text))
        elif entity_code == FebEntity.BOOK:
            utt.entities.append(FebBook(tokens=entity_text))
        else:
            utt.add_error(FebError(FebError.ET_INP_DATA, self, {FebError.EC_DATA_VAL: entity_text}))
        return  utt

    # don't override basic realization
    # def pack_result(self, utt, ret_obj_l):



    # def _splitter(self, pars_str):
    #     log.info(f'feb_t1_parser _splitter pars_str={pars_str}')
    #     res = {}
    #     qnl = re.findall(r'^(\w+)', pars_str)
    #     if len(qnl) > 0:
    #         res['query_name'] = qnl[0]
    #     else:
    #         return {'error': 'question_type_not_found'}
    #     res['nent_lst'] = [{'nent_type': nent_type, 'nent_str': nent_str}
    #                        for nent_type, nent_str in re.findall(r'(?:<(.+?):(.+?)>)', pars_str)]
    #     log.info(f'feb_t1_parser _splitter res={res}')
    #     return res
