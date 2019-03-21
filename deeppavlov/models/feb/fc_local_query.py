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

from question2wikidata import questions_ldb as questions
from question2wikidata import functions_ldb as functions
# from question2wikidata.server_queries import queries
from question2wikidata.local_queries import queries
log = get_logger(__name__)


@register('local_query')
class LocalQuery(FebComponent):
    """Convert batch of strings
      """

    @classmethod
    def component_type(cls):
        return cls.INTERMEDIATE_COMPONENT

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def test_and_prepare(self, utt: FebUtterance):
        """
        Test input data and prepare data to process
        :param utt: FebUtterance
        :return: list(tuple(FebIntent, {})) - for FebIntent context is void
        """
        return [(i, {'entities': utt.entities}) for i in utt.intents]

    def process(self, intent: FebIntent, context):
        """
        Setting qid for entity
        :param entity: FebEntity
        :param context: void dict
        :return: None (all results saved in place (for arguments))
        """
        entities = context['entities'] # list of FebEntity
        print(entities)
        if not intent.has_errors():
            # algorithm support only one parameter of certain type!
            query = queries[intent.type]['query']
            print(query)
            query_params = {}
            errors = False
            for param_name, param_type in queries[intent.type]['params'].items():
                ent_l = [ent for ent in entities if ent.type == param_type]
                # disambiguation problem!
                # filtering ent with qid != None:
                qid_l = [ent.qid for ent in ent_l if ent.qid]
                id_l = [ent.id for ent in ent_l if ent.id]
                print(qid_l)
                print(id_l)
                if len(qid_l) == 1 and len(id_l) == 1:
                    query_params[param_name] = (qid_l[0], id_l[0])
                elif len(qid_l) == 0 and len(id_l) == 1:
                    query_params[param_name] = (None, id_l[0])
                elif len(qid_l) == 1 and len(id_l) == 0:
                    query_params[param_name] = (qid_l[0], None)

                elif len(qid_l) == 0 and len(id_l) == 0:
                    intent.add_error(FebError(FebError.ET_INP_DATA, self, {FebError.EC_DATA_NONE:
                                                                               {'param_name': param_name,
                                                                                'param_type': param_type}}))
                    errors = True
                else:
                    intent.add_error(FebError(FebError.ET_INP_DATA, self, {FebError.EC_DATA_DISABIG:
                                                                               {'param_name': param_name,
                                                                                'param_type': param_type,
                                                                                'entities': (ent_l,id_l)}}))
                    errors = True
            print('qparams', query_params)
            #TODO: продумать, как обойти ent_l[0]
            if not errors:
                if intent.type.startswith('book_') and len(entities) > 0:
                    entities_list = [e for e in entities if isinstance(e, FebBook)]
                else:
                    entities_list = [e for e in entities if isinstance(e, FebAuthor)]
                ent = entities_list[0]

                try:
                    intent.result_val = {intent.type : ent.info[intent.type]}
                except KeyError:
                    intent.result_val = {'error':'error not in entity.info'}
                # intent.result_val = functions.execute_query_sql(query, **query_params)

            # intent.result_str = str()
        return intent

    def pack_result(self, utt: FebUtterance, ret_obj_l):
        """
        Trivial packing
        :param utt: current FebUtterance
        :param ret_obj_l: list of entities
        :return: utt with list of updated intents
        """
        var_dump(header='wikidata_query pack_result', msg = f'utt = {utt}, ret_obj_l = {ret_obj_l}')
        utt.intents = ret_obj_l

        if 'error' in utt.get_gen_context()['results_keys'] or None in utt.get_gen_context()['results']:
        # if len(utt.intents[0].errors):
            return FebStopBranch.STOP, [utt]
        else:
            return [utt], FebStopBranch.STOP


    # @overrides
    # def __call__(self, batch, *args, **kwargs):
    #     for utt in batch:
    #         utt_type = utt.get(Utterance.TYPE.value)
    #         ne_l = utt.get(Utterance.NAMED_ENTITY_LST.value)
    #         res_type, res_val = self._make_query(utt_type, ne_l)
    #         utt[res_type] = res_val
    #         # for ne in ne_l:
    #         #     qid = self._make_query(utt_type,
    #         #                                  ne.get(NamedEntity.NE_TYPE.value))
    #         #     if qid:
    #         #         ne[NamedEntity.NE_QID.value] = qid
    #         #     else:
    #         #         utt[Utterance.ERROR.value] = UtteranceErrors.QID_NOT_FOUND.value
    #         #         utt.get(Utterance.ERROR_VAL_LST.value, list()).append(ne)
    #     return batch

    # def _make_query(self, query, ne_l):
    #     log.info(f'wikiddata_query _make_query query={query}, ne_l={ne_l}')
    #     try:
    #         if query in queries:
    #             qparams = {}
    #             for param_name, param_type in queries[query]['params'].items():
    #                 params = [e[NamedEntity.NE_QID.value] for e in ne_l if e[NamedEntity.NE_TYPE.value] == param_type]
    #                 if len(params) == 1:
    #                     qparams[param_name] = params[0]
    #                 else:
    #                     return Utterance.ERROR.value, UtteranceErrors.WIKIDATA_QUERY_ERROR.value  # TODO: Error!
    #
    #             qr = queries[query]['query']
    #             log.info(f'wikiddata_query _make_query query_name={query}, qparams={qparams}, query={qr}')
    #             query_result = functions.execute_query(qr, **qparams)
    #             log.info(f'wikiddata_query _make_query query_result={query_result}')
    #             # return Utterance.ERROR.value, UtteranceErrors.WIKIDATA_QUERY_ERROR.value
    #             return Utterance.RESULT.value, query_result
    #         else:
    #             return Utterance.ERROR.value, UtteranceErrors.WIKIDATA_QUERY_ERROR.value  # TODO: Error!
    #     except:
    #         return Utterance.ERROR.value, UtteranceErrors.WIKIDATA_QUERY_ERROR.value  # TODO: Error!
