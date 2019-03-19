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


from .feb_objects import *
from .feb_common import FebComponent
from ..feb import CONTEXTS

from datetime import datetime


log = get_logger(__name__)


@register('fc_memorize_context')
class FebMemorizeContext(FebComponent):
    """Convert utt to strings
      """
    @classmethod
    def component_type(cls):
        return cls.FINAL_COMPONENT

    def __init__(self, memorize: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.memorize = memorize

    # don't override basic realization
    # def test_and_prepare(self, utt):
    def test_and_prepare(self, utt):
        
        var_dump(header='memorize_context', msg='memorize_context started!')

        if self.memorize:
            chat_id = utt.chat_id
            current_timestamp = datetime.now().timestamp()

            entities = utt.entities
            intent = utt.alt_ans_pattern

            CONTEXTS[chat_id] = {'timestamp': current_timestamp, 'prev_entities': entities, 'alt_intent': intent}
        else:
            if utt.chat_id in CONTEXTS:
                CONTEXTS.pop(utt.chat_id)

        return [(utt, {})]