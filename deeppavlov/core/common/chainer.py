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

import pickle
from typing import Union, Tuple, List, Dict

from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.nn_model import NNModel
from deeppavlov.core.models.serializable import Serializable

from deeppavlov.models.feb.feb_common import FebComponent
from deeppavlov.models.feb.feb_objects import FebStopBranch

# class FebStopBranch(object):
#     STOP = 'FebStopBranch.STOP'
#     CONTINUE = 'FebStopBranch.CONTINUE'
#     pass


def var_dump(msg, header=''): print(f'-----{header}-----\n{msg}\n----\n')

class Chainer(Component):
    """
    Builds an agent/component pipeline from heterogeneous components (Rule-based/ML/DL). It allows to train
    and infer models in a pipeline as a whole.

    Attributes:
        pipe: list of components and their input and output variable names for inference
        train_pipe: list of components and their input and output variable names for training and evaluation
        in_x: names of inputs for pipeline inference mode
        out_params: names of pipeline inference outputs
        in_y: names of additional inputs for pipeline training and evaluation modes
        forward_map: list of all variables in chainer's memory after  running every component in ``self.pipe``
        train_map: list of all variables in chainer's memory after  running every component in ``train_pipe.pipe``
        main: reference to the main component

    Args:
        in_x: names of inputs for pipeline inference mode
        out_params: names of pipeline inference outputs
        in_y: names of additional inputs for pipeline training and evaluation modes
    """
    def __init__(self, in_x: Union[str, list] = None, out_params: Union[str, list] = None,
                 in_y: Union[str, list] = None, *args, **kwargs) -> None:
        self.pipe: List[Tuple[Tuple[List[str], List[str]], List[str], Component]] = []
        self.train_pipe = []
        if isinstance(in_x, str):
            in_x = [in_x]
        if isinstance(in_y, str):
            in_y = [in_y]
        if isinstance(out_params, str):
            out_params = [out_params]
        self.in_x = in_x or ['x']
        self.in_y = in_y or ['y']
        self.out_params = out_params or self.in_x

        self.forward_map = set(self.in_x)
        self.train_map = self.forward_map.union(self.in_y)

        self.main = None

    def append(self, component: Component, in_x: [str, list, dict]=None, out_params: [str, list]=None,
               in_y: [str, list, dict]=None, main=False):
        if isinstance(in_x, str):
            in_x = [in_x]
        if isinstance(in_y, str):
            in_y = [in_y]
        if isinstance(out_params, str):
            out_params = [out_params]
        in_x = in_x or self.in_x
        if isinstance(in_x, dict):
            x_keys, in_x = zip(*in_x.items())
        else:
            x_keys = []
        out_params = out_params or in_x
        if in_y is not None:
            if isinstance(in_y, dict):
                y_keys, in_y = zip(*in_y.items())
            else:
                y_keys = []
            keys = x_keys + y_keys

            if bool(x_keys) != bool(y_keys):
                raise ConfigError('`in` and `in_y` for a component have to both be lists or dicts')

            component: NNModel
            main = True
            assert self.train_map.issuperset(in_x+in_y), ('Arguments {} are expected but only {} are set'
                                                          .format(in_x+in_y, self.train_map))
            preprocessor = Chainer(self.in_x, in_x+in_y, self.in_y)
            for (t_in_x_keys, t_in_x), t_out, t_component in self.train_pipe:
                if t_in_x_keys:
                    t_in_x = dict(zip(t_in_x_keys, t_in_x))
                preprocessor.append(t_component, t_in_x, t_out)

            def train_on_batch(*args, **kwargs):
                preprocessed = preprocessor.compute(*args, **kwargs)
                if len(in_x+in_y) == 1:
                    preprocessed = [preprocessed]
                if keys:
                    return component.train_on_batch(**dict(zip(keys, preprocessed)))
                else:
                    return component.train_on_batch(*preprocessed)

            self.train_on_batch = train_on_batch
            self.process_event = component.process_event
        if main:
            self.main = component
        if self.forward_map.issuperset(in_x):
            self.pipe.append(((x_keys, in_x), out_params, component))
            self.forward_map = self.forward_map.union(out_params)

        if self.train_map.issuperset(in_x):
            self.train_pipe.append(((x_keys, in_x), out_params, component))
            self.train_map = self.train_map.union(out_params)
        else:
            raise ConfigError('Arguments {} are expected but only {} are set'.format(in_x, self.train_map))

    def compute(self, x, y=None, targets=None):
        if targets is None:
            targets = self.out_params
        in_params = list(self.in_x)
        if len(in_params) == 1:
            args = [x]
        else:
            args = list(zip(*x))

        if y is None:
            pipe = self.pipe
        else:
            pipe = self.train_pipe
            if len(self.in_y) == 1:
                args.append(y)
            else:
                args += list(zip(*y))
            in_params += self.in_y

        return self._compute(*args, pipe=pipe, param_names=in_params, targets=targets)

    def __call__(self, *args):
        return self._compute(*args, param_names=self.in_x, pipe=self.pipe, targets=self.out_params)

    @staticmethod
    def _compute(*args, param_names, pipe, targets):
        #Срабатывает после отправки сообщения из TG
        var_dump(header='Параметры на входе', msg = f'args = {args},\n param_names={param_names},\n pipe={pipe},\n targets={targets}\n')
        # expected = set(targets)
        final_pipe = []
        for (in_keys, in_params), out_params, component in reversed(pipe):
            if not isinstance(component, FebComponent):
                raise RuntimeError(f'Поддерживаются только компоненты FebComponent')
                # break
            else:
                final_pipe.append(((in_keys, in_params), out_params, component))
            # if expected.intersection(out_params):
            #     expected = expected - set(out_params) | set(in_params)
            #     final_pipe.append(((in_keys, in_params), out_params, component))
            #     var_dump(header='_COMPUTE', msg=f'Component {component} added!')
            # else:
            #     var_dump(header='_COMPUTE', msg=f'Component {component} not added!')
        final_pipe.reverse()
        # if not expected.issubset(param_names):
        #     raise RuntimeError(f'{expected} are required to compute {targets} but were not found in memory or inputs')
        pipe = final_pipe

        mem = dict(zip(param_names, args))
        del args
        var_dump(header = '_COMPUTE', msg=f'Final PIPE = {pipe}')

        chainer_iter_num = 1

        components_done = []

        while not set(targets).issubset(set(mem.keys())): # пока не получим результат выходного параметра
            for (in_keys, in_params), out_params, component in pipe:
                #В начале выполнения каждого компонента здесь:
                # in_keys = []
                # in_params = 'in' из конфига
                # out_params = 'out' из конфигаа
                # component - ссылка на объект компонента

                component_ready = set(in_params).issubset(set(mem.keys())) and component not in components_done 
                #если все входные параметры в памяти и компонент еще не отрабатывал
                if component_ready:
                    x = [mem[k] for k in in_params]
                    var_dump(header=f'[{chainer_iter_num}] In Pipe Loop Start', msg = f'(in_keys={in_keys},\n in_params={in_params})\n, out_params={out_params},\n component={component}\n, mem = {mem.keys()}, \n x = {x}')
                    res = component(*x)
                    if len(out_params) == 1:
                        if res != FebStopBranch.STOP:
                            mem[out_params[0]] = res 
                    else:
                        for out_param_key, out_param_value in zip(out_params, res):
                            if out_param_value != FebStopBranch.STOP:
                                mem[out_param_key] = out_param_value
                    components_done.append(component)
                    var_dump(header=f'[{chainer_iter_num}] In Pipe Loop End', msg = f'(in_keys={in_keys},\n in_params={in_params})\n, out_params={out_params},\n component={component}\n, x={x}, \nmem={mem.keys()}, \nres={res}')
                    chainer_iter_num += 1
                    break
            else:
                raise RuntimeError(f'Не хватает компонент, чтобы вычислить {targets}')

                
        #Необходимо разделять статус компонент для разных пользователей
        '''for (in_keys, in_params), out_params, component in pipe:
            #В начале выполнения каждого компонента здесь:
            # in_keys = []
            # in_params = 'in' из конфига
            # out_params = 'out' из конфигаа
            # component - ссылка на объект компонента
            var_dump(header='In Pipe Loop Start', msg = f'(in_keys={in_keys},\n in_params={in_params})\n, out_params={out_params},\n component={component}\n, mem = {mem}')
            
            x = [mem[k] for k in in_params]
            if in_keys:
                res = component(**dict(zip(in_keys, x)))
            else:
                if FebStopBranch.STOP not in x:
                    res = component(*x)
                    if len(out_params) == 1:
                        mem[out_params[0]] = res
                    else:
                        mem.update(zip(out_params, res))
                    var_dump(header='In Pipe Loop End', msg = f'res={res}')  
                    var_dump(header='In Pipe Loop End', msg = f'(in_keys={in_keys},\n in_params={in_params})\n, out_params={out_params},\n component={component}\n, x={x}, \nmem={mem}')

                else:
                    if len(out_params) == 1 and (out_params[0] not in mem):
                        mem[out_params[0]] = FebStopBranch.STOP
                    else:
                        for out_param in out_params:
                            if out_param not in mem:
                                mem[out_param] = FebStopBranch.STOP
                    var_dump(header='In Pipe Loop End', msg = f'Component {component} skipped!')      
                    continue

            if len(out_params) == 1:
                mem[out_params[0]] = res
            else:
                mem.update(zip(out_params, res))'''
            
        res = [mem[k] for k in targets]
        if len(res) == 1:
            res = res[0]
        # var_dump(header='Параметры на выходе', msg = f'param_names={param_names}\n, pipe={pipe}\n, targets={targets}\n, res={res}\n')
        return res

    def get_main_component(self) -> Serializable:
        return self.main or self.pipe[-1][-1]

    def save(self) -> None:
        self.get_main_component().save()

    def load(self) -> None:
        for in_params, out_params, component in self.train_pipe:
            if callable(getattr(component, 'load', None)):
                component.load()

    def reset(self) -> None:
        for in_params, out_params, component in self.train_pipe:
            if callable(getattr(component, 'reset', None)):
                component.reset()

    def destroy(self):
        for in_params, out_params, component in self.train_pipe:
            if callable(getattr(component, 'destroy', None)):
                component.destroy()
        self.pipe.clear()
        self.train_pipe.clear()

    def serialize(self) -> bytes:
        data = []
        for in_params, out_params, component in self.train_pipe:
            data.append(component.serialize())
        return pickle.dumps(data)

    def deserialize(self, data: bytes) -> None:
        data = pickle.loads(data)
        for in_params, out_params, component in self.train_pipe:
            component.deserialize(data)
