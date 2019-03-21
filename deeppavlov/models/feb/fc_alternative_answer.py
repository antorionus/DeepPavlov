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
import answers
from patterns import PATTERNS
from alternative_matrix import alternate_matrix


from .feb_objects import *
from .feb_common import FebComponent


log = get_logger(__name__)


@register('fc_alternative_answer')
class FebAlternativeAnswer(FebComponent):
    """Convert utt to strings
      """

    @classmethod
    def component_type(cls):
        return cls.INTERMEDIATE_COMPONENT

    def __init__(self, template=None, **kwargs):
        super().__init__(**kwargs)
        self.template = template # Шаблон для ответа 
    # don't override basic realization
    # def test_and_prepare(self, utt):

    @staticmethod
    def alternative_intent_select(info_dict, current_intent):
        info_keys_list = list(info_dict.keys())
        score_dict = {}
        for known_info_key in info_keys_list:
            if known_info_key in alternate_matrix.keys():
                score = alternate_matrix[current_intent][known_info_key]
                score_dict[score] = known_info_key
            else:
                continue
        if len(score_dict) > 0:
            print(f"___alternative_intent_select____ chosen new alternate: {score_dict[max(score_dict)]}")
            return score_dict[max(score_dict)]
        else:
            print(f"___alternative_intent_select____ : No intent selected (return None)")
            return None

    def pack_result(self, utt: FebUtterance, ret_obj_l):
        """
        Trivial packing
        :param utt: current FebUtterance
        :param ret_obj_l: list of entities
        :return: utt with list of updated intents
        """
        var_dump(header='FebAlternativeAnswer pack_result', msg = f'utt = {utt}, ret_obj_l = {ret_obj_l}')

        #логика предложения альтернативных вариантов
        intents_list = [intent for intent in utt.intents]
        if len(intents_list) > 0 and len(utt.entities) > 0:
            intent = intents_list[0]
            if intent.type.startswith('book_'):
                entities_list = [e for e in utt.entities if isinstance(e, FebBook)]
                ent = entities_list[0]
            else:
                entities_list = [e for e in utt.entities if isinstance(e, FebAuthor)]
                ent = entities_list[0]

            if ent.info:
                alt_pattern = self.alternative_intent_select(ent.info, intent.type)
                setattr(utt, 'alt_ans_pattern', alt_pattern)
            else:
                setattr(utt, 'alt_ans_pattern', None)
        else:
            setattr(utt, 'alt_ans_pattern', None)
        # TEMP_CAN_SUGGEST_ALTERNATIVE = 'TEST_ALTERNATIVE' in utt.text #False True

        # utt.alt_ans_pattern
        if not utt.alt_ans_pattern:
            return FebStopBranch.STOP, [utt]
        else:
            return [utt], FebStopBranch.STOP

        # if not TEMP_CAN_SUGGEST_ALTERNATIVE:
        #     return FebStopBranch.STOP, [utt]
        # else:
        #     return [utt], FebStopBranch.STOP
        
        # return utt, FebStopBranch.STOP