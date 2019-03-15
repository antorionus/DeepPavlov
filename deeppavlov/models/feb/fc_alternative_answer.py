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


    def pack_result(self, utt: FebUtterance, ret_obj_l):
        """
        Trivial packing
        :param utt: current FebUtterance
        :param ret_obj_l: list of entities
        :return: utt with list of updated intents
        """
        var_dump(header='FebAlternativeAnswer pack_result', msg = f'utt = {utt}, ret_obj_l = {ret_obj_l}')

        #логика предложения альтернативных вариантов
        TEMP_CAN_SUGGEST_ALTERNATIVE = 'TEST_ALTERNATIVE' in utt.text

        
        if not TEMP_CAN_SUGGEST_ALTERNATIVE:
            return FebStopBranch.STOP, [utt]
        else:
            return [utt], FebStopBranch.STOP
        
        # return utt, FebStopBranch.STOP