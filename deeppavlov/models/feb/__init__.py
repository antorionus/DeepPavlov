
__all__ = ['feb_common', 'nent_to_qent', 't1_parser', 't1_text_generator']


#Предзагрузка для модуля NER
from question2wikidata.extractors import title
BOOK_NAMES_EXTRACTOR = title.titleExtractor()

#Файл для логирования
LOG_FILE = open('LOG_FILE.jsonl', 'a', encoding = 'utf-8')

#Режим отладки
DEBUG = False


#Словарь контекстов, ключ - chat_id, значение - ???
CONTEXTS = dict()
MAX_REMEMBER_TIME = 10 #в секундах