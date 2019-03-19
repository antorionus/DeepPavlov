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
from collections import OrderedDict


from .feb_objects import *
from .feb_common import FebComponent


log = get_logger(__name__)


@register('fc_error_analysis')
class FebErrorAnalysis(FebComponent):
    """Convert utt to strings
      """
    @classmethod
    def component_type(cls):
        return cls.INTERMEDIATE_COMPONENT

    def __init__(self, error_place=None, **kwargs):
        super().__init__(**kwargs)
        self.error_place = error_place # место, где TextGen будет искать постфикс для шаблона в случае ошибки
    # don't override basic realization
    # def test_and_prepare(self, utt):



    def process(self, utt: FebUtterance, context):
        """
        Main processing function
        :param obj: obj to process
        :param context: dict with processing context
        :return: processed object
        """

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
        # Подготовка данных для шаблонизатора

        # TODO: перенести эту логику в отдельный компонент ErrorAnalysis
        # TODO: создать файлы с тематическими шаблонами, подгружать с помощью load_path ?


        gen_context = utt.get_gen_context()

        query_type = gen_context['query_name']
        params_list = gen_context['params']
        results_dict = gen_context['results']

        possible_options = ['no_class', 'nth_but_class', 'no_qid', 'no_data']
        answer = OrderedDict((o, FebStopBranch.STOP) for o in possible_options)

        prefix, pattern_type = None, '_'

        if query_type == 'intent_not_set_type' or query_type == 'unsupported_type':
            if len(params_list) != 0:
                answer['no_class'] = [utt]
                pattern_type += params_list[0].type
            else:
                answer['no_class'] = [utt]
                pattern_type = ''
        else:
            if len(params_list) == 0:
                answer['nth_but_class'] = [utt]
                pattern_type += query_type
            elif results_dict['error'] == 'DataNotFound': #когда это отрабатывает?
                answer['no_data'] = [utt] 
                pattern_type += query_type
            elif len(params_list) != 0:
                qid = params_list[0].qid
                if qid is None:
                    answer['no_qid'] = [utt]
                    pattern_type += query_type
                else:
                    answer['no_data'] = [utt]
                    pattern_type += query_type

        exec(f"{self.error_place} = '{pattern_type}'")
        # utt.pattern_type = pattern_type
        var_dump(header='fc_error_analysis', msg = f'{tuple(answer.values())}')
        return tuple(answer.values())




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


