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


@register('feb_text_generator')
class FebSideTextGenerator(FebComponent):
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


    def process(self, utt: FebUtterance, context):
        """
        Main processing function
        :param obj: obj to process
        :param context: dict with processing context
        :return: processed object
        """
        # Подготовка данных для шаблонизатора

        # TODO: создать файлы с тематическими шаблонами, подгружать с помощью load_path ?

        gen_context = utt.get_gen_context()

        # self.template + +  utt.als_ans

        # if false : self.template = have_alternative or no_alternative
        result = answers.answer(gen_context, prepared_pattern = self.template+'_'+utt.alt_ans_pattern if utt.alt_ans_pattern else self.template)
        side_utt = FebUtterance(f'{result}')
        var_dump(header='textgen', msg=f'fc_side_text_gen вернул {side_utt}')
        return side_utt


    def pack_result(self, utt, ret_obj_l):
        """
        Packing results of processing
        :param utt: current FebUtterance
        :param ret_obj_l: list of processed objects
        :return: utt with added values from ret_obj_l
        """
        # basic realization:
        # doesn't updated utt
        # assert utt is ret_obj_l[0], 'Basic realization of pack_result() is incorrect!'
        return ret_obj_l[0]




c_ = {
    'query_name': 'author_genres',
    'params': [
        {'type': 'author',
        'text': 'Достоевского',
        'normal_form': 'Достоевский',
        'qid': 'Q1234',
        'text_from_base': 'Федор Михайлович Достоевский'},
        {'type': 'author',
         'name_from_text': 'Пушкина',
         'name_normal_form': 'Пушки',
         'qid': 'Q1234',
         'name_from_base': 'Александр Сергеевич Пушкин'},
        {'type': 'book',
         'name_from_text': 'войны и мира',
         'name_normal_form': 'война и мир',
         'qid': 'Q1234',
         'name_from_base': 'Война и мир'},
        {'type': 'book',
         'name_from_text': 'Медного всадника',
         'name_normal_form': 'Медный всадник',
         'qid': 'Q1234',
         'name_from_base': 'Медный всадник'}
    ],
    'results': [
        {'error': 'DataNotFound'}]
}

a_ = """
a = {
    'params': {
        'query_name': 'author_genres',
        'author_name': 'Толстой'
    },
    'results': [
        {'genreLabel': 'роман'},
        {'genreLabel': 'драматическая форма'},
        {'genreLabel': 'рассказ'},
        {'genreLabel': 'повесть'}]
"""


    # don't override basic realization
    # def pack_result(self, utt, ret_obj_l):


