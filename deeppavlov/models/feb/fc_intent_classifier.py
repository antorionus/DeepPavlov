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

# from .feb_common import NamedEntity, NamedEntityType, Utterance, UtteranceErrors

from .feb_objects import *
from .feb_common import FebComponent

from question2wikidata import questions, functions
from intention_classifier import predict


log = get_logger(__name__)


@register('fc_intent_classifier')
class IntentClassifier(FebComponent):
    """Convert batch of strings
      """

    QUESTION_WORDS = {
        'как': False,
        'где': False,
        'куда': False,
        'откуда': False,
        'когда': False,
        'почему': False,
        'зачем': False,
        'отчего': False,
        'кто': False,
        'что': False,
        'какой': False,
        'чей': False,
        'сколько': False,
        'каков': False,
        'который': False
    }

    @classmethod
    def component_type(cls):
        return cls.INTERMEDIATE_COMPONENT

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def test_and_prepare(self, utt: FebUtterance):
        """
        Test input data and prepare data to process
        :param utt: FebUtterance
        :return: list(tuple(FebEntity, {})) - for FebEntity context is void
        """
        var_dump(header = 'intent_classifier', msg = utt)
        return [(utt, {
                    'tokens': utt.tokens,
                    'entities': utt.entities
                      })]

    def process(self, obj, context):
        """
        Setting qid for entity
        :param entity: FebEntity
        :param context: void dict
        :return: None (all results saved in place (for arguments))
        """
        utt, tokens, entities = obj, context['tokens'], context['entities']
        tokens_for_classifier = []
        question_words = dict(self.QUESTION_WORDS)
        for token in tokens:
            if token.type == FebToken.NOT_PUNCTUATION and token.normal_form in IntentClassifier.QUESTION_WORDS:
                question_words[token.normal_form] = True
            if (token.type == FebToken.NOT_PUNCTUATION) and (FebToken.TAG_AUTHOR not in token.tags) and\
                (FebToken.TAG_BOOK not in token.tags):
                tokens_for_classifier.append(f'{token.normal_form}_{token.pos}')
        
        author_count, book_count = 0, 0

        var_dump(header='intent_classifier entities', msg = entities)
        for entity in entities:
            if isinstance(entity, FebAuthor): author_count += 1
            if isinstance(entity, FebBook): book_count += 1

        counts = {'author_count': author_count, 'book_count': book_count}
        # prepared_data = [tokens_for_classifier, {'author_count': author_count, 'book_count': book_count}]

        question_words_param = [question_words[key] for key in sorted(question_words.keys())]
        log.debug(f'\nquestion_words:{question_words}, question_words_param:{question_words_param}\n')
        intent = predict(tokens_for_classifier, question_words=question_words_param, **counts)
        var_dump(header='intent_classifier intent', msg = intent)

        intents_code = intent
        if FebIntent.in_supported_types(intents_code):
            intent = FebIntent(intents_code)
        else:
            intent = FebIntent(FebIntent.UNSUPPORTED_TYPE)
            intent.add_error(FebError(FebError.ET_INP_DATA, self, {FebError.EC_DATA_VAL: intents_code}))
        utt.intents.append(intent)
        # var_dump(header = 'intent_classifier', msg = prepared_data)

        return utt


    def pack_result(self, utt: FebUtterance, ret_obj_l):
        """
        Trivial packing
        :param utt: current FebUtterance
        :param ret_obj_l: list of entities
        :return: utt with list of updated entities
        """
        var_dump(header='intent_classifier pack_result', msg = f'utt = {utt}, ret_obj_l = {ret_obj_l}')
        if utt.intents[0].type == FebIntent.UNSUPPORTED_TYPE:
            return FebStopBranch.STOP, ret_obj_l
        else:
            return ret_obj_l, FebStopBranch.STOP



    # @overrides
    # def __call__(self, batch, *args, **kwargs):
    #     for utt in batch:
    #         ne_l = utt.get(Utterance.NAMED_ENTITY_LST.value)
    #         for ne in ne_l:
    #             qid = self._extract_entities(ne.get(NamedEntity.NE_STRING.value),
    #                                          ne.get(NamedEntity.NE_TYPE.value))
    #             if qid:
    #                 ne[NamedEntity.NE_QID.value] = qid
    #             else:
    #                 utt[Utterance.ERROR.value] = UtteranceErrors.QID_NOT_FOUND.value
    #                 utt.get(Utterance.ERROR_VAL_LST.value, list()).append(ne)
    #     return batch
    #
    #
    # def _extract_entities(self, ne_str, param_type):
    #     log.info(f'nent_to_qent _extract_entities query={ne_str}, param_type={param_type}')
    #     qid = functions.get_qid(ne_str, param_type)
    #     log.info(f'nent_to_qent _extract_entities qid={qid}')
    #     return qid

