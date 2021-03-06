"""
Copyright 2017 Neural Networks and Deep Learning lab, MIPT

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from pathlib import Path

import telebot
from telebot import apihelper

from deeppavlov.agents.default_agent.default_agent import DefaultAgent
from deeppavlov.agents.processors.default_rich_content_processor import DefaultRichContentWrapper
from deeppavlov.core.agent import Agent
from deeppavlov.core.agent.rich_content import RichMessage
from deeppavlov.core.commands.infer import build_model
from deeppavlov.core.common.file import read_json
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.common.paths import get_settings_path
from deeppavlov.skills.default_skill.default_skill import DefaultStatelessSkill


log = get_logger(__name__)

SERVER_CONFIG_FILENAME = 'server_config.json'
TELEGRAM_MODELS_INFO_FILENAME = 'models_info.json'


def init_bot_for_model(agent: Agent, token: str, model_name: str, proxy_key: str=None, proxy_val: str=None):

    models_info_path = Path(get_settings_path(), TELEGRAM_MODELS_INFO_FILENAME).resolve()
    models_info = read_json(str(models_info_path))
    model_info = models_info[model_name] if model_name in models_info else models_info['@default']

# proxy added:
#   apihelper.proxy = {'http':'http://10.10.1.10:3128'}
#   IP: deimos.public.opennetwork.cc
#   1090
#   Login: 333744069
#   Password: eMEKzCQu
#    apihelper.proxy={'https':'socks5://333744069:eMEKzCQu@deimos.public.opennetwork.cc:1090'}		
#
# from: DeepPavlov\utils\settings\server_config.json
    if  proxy_key and proxy_val:
        apihelper.proxy = {proxy_key : proxy_val}
    bot = telebot.TeleBot(token)

    @bot.message_handler(commands=['start'])
    def send_start_message(message):
        chat_id = message.chat.id
        out_message = model_info['start_message']
        bot.send_message(chat_id, out_message)

    @bot.message_handler(commands=['help'])
    def send_help_message(message):
        chat_id = message.chat.id
        out_message = model_info['help_message']
        bot.send_message(chat_id, out_message)

    @bot.message_handler()
    def handle_inference(message):
        chat_id = message.chat.id
        context = message.text
        bot.send_chat_action(chat_id, 'typing')
        response: RichMessage = agent([context], [chat_id])[0]
        for message in response.json():
            message_text = message['content']
            bot.send_message(chat_id, message_text)

    bot.polling()


def interact_model_by_telegram(config, token=None, proxy_key=None, proxy_val=None):
    server_config_path = Path(get_settings_path(), SERVER_CONFIG_FILENAME)
    server_config = read_json(server_config_path)
    token = token if token else server_config['telegram_defaults']['token']
# from: DeepPavlov\utils\settings\server_config.json
    proxy_key = proxy_key if proxy_key else server_config['telegram_defaults']['proxy_key']
    proxy_val = proxy_val if proxy_val else server_config['telegram_defaults']['proxy_val']
    if not token:
        e = ValueError('Telegram token required: initiate -t param or telegram_defaults/token '
                       'in server configuration file')
        log.error(e)
        raise e

    model = build_model(config)
    model_name = type(model.get_main_component()).__name__
    skill = DefaultStatelessSkill(model)
    agent = DefaultAgent([skill], skills_processor=DefaultRichContentWrapper())
    init_bot_for_model(agent, token, model_name, proxy_key, proxy_val)
