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
from ..feb import CONTEXTS, MAX_REMEMBER_TIME

from question2wikidata import questions, functions
from datetime import datetime

log = get_logger(__name__)


@register('fc_recall_context')
class FebRecallContext(FebComponent):
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
        

        return [(utt, {})]

    def pack_result(self, utt, ret_obj_l):

        var_dump(header='recall_context', msg='recall_context started!')

        chat_id: str = utt.chat_id
        context = CONTEXTS.get(chat_id, None)
        now = datetime.now().timestamp()

        if context is not None and (now - context['timestamp']  > MAX_REMEMBER_TIME):
            CONTEXTS.pop(chat_id, None)
            context = None

        var_dump(header='recall_context', msg=f'current context = {context}!')
        var_dump(header='recall_context', msg=f'all contexts = {CONTEXTS}!')
        
        var_dump(header='RecallContext', msg=f'ret_obj_l={ret_obj_l}, FebStopBranch() = {FebStopBranch()}')

        if chat_id not in CONTEXTS:
            utt.context = {}
            return ret_obj_l, FebStopBranch.STOP
        else:
            utt.context = CONTEXTS[utt.chat_id]
            return FebStopBranch.STOP, ret_obj_l