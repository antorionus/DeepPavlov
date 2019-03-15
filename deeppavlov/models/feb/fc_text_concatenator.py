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
from time import time


log = get_logger(__name__)


@register('fc_text_concatenator')
class FebTextConcatenator(FebComponent):
    """Convert utt to strings
      """

    @classmethod
    def component_type(cls):
        return cls.INTERMEDIATE_COMPONENT

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @overrides
    def __call__(self, batch, *args, **kwargs):
        # Переопределяем __call__ для корректной работы 
        # В batch в таком виде попадает только 1 элемент, остальные попадают в args
        # Считаем, что в batch - utt со всей накопленной информацией
        # Сообщения в args - utt's с заполненным текстом

        start = time()
        batch = list(batch)
        for _ in args: batch.extend(_)

        var_dump(header=f'__call__ batch in {self.__class__.__name__}', msg=f'batch = {batch}, \n args = {args}, \n kwargs={kwargs}')
        main_utt, *add_msgs = batch
        var_dump(header=f'__call__ ', msg=f'main_utt = {main_utt}, \n add_msgs = {add_msgs}, \n ')

        re_text = main_utt.re_text
        for add_msg in add_msgs:
            re_text += ' ' + add_msg.text

        main_utt.re_text = re_text


        finish = time()
        taken = finish - start
        FebComponent.TIME_TAKEN += taken
        var_dump(header = f'Компонент {self}', msg = f'Выполнение заняло {taken}. Выполняется уже: {FebComponent.TIME_TAKEN}')
        return [main_utt]


    # don't override basic realization
    # def test_and_prepare(self, utt):


    # def pack_result(self, utt: FebUtterance, ret_obj_l):
    #     """
    #     Trivial packing
    #     :param utt: current FebUtterance
    #     :param ret_obj_l: list of entities
    #     :return: utt with list of updated intents
    #     """
    #     var_dump(header='FebAlternativeAnswer pack_result', msg = f'utt = {utt}, ret_obj_l = {ret_obj_l}')

    #     #логика предложения альтернативных вариантов
    #     TEMP_CAN_SUGGEST_ALTERNATIVE = 'TEST_ALTERNATIVE' in utt.text

        
    #     if not TEMP_CAN_SUGGEST_ALTERNATIVE:
    #         return FebStopBranch.STOP, [utt]
    #     else:
    #         return [utt], FebStopBranch.STOP
        
    #     # return utt, FebStopBranch.STOP