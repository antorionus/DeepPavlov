from datetime import datetime
import uuid

from deeppavlov.core.agent_v2.state_schema import User, Human, Utterance, BotUtterance, Dialog
from deeppavlov.core.agent_v2.connection import state_storage
from deeppavlov.core.agent_v2.bot import BOT

########################### Test case #######################################

# User.drop_collection()
# Human.drop_collection()

Dialog.objects.delete()
Utterance.objects.delete()
BotUtterance.objects.delete()
# DialogHistory.objects.delete()
# User.objects.delete()
# Human.objects.delete()


h_user = Human(user_telegram_id=uuid.uuid4())
b_user = User(user_type='bot')

h_utt_1 = Utterance(text='Привет!', user=h_user, date_time=datetime.utcnow())
b_utt_1 = BotUtterance(text='Привет, я бот!', user=b_user, active_skill='chitchat',
                       confidence=0.85, date_time=datetime.utcnow())

h_utt_2 = Utterance(text='Как дела?', user=h_user, date_time=datetime.utcnow())
b_utt_2 = BotUtterance(text='Хорошо, а у тебя как?', user=b_user,
                       active_skill='chitchat',
                       confidence=0.9333, date_time=datetime.utcnow())

h_utt_3 = Utterance(text='И у меня нормально. Когда родился Петр Первый?', user=h_user, date_time=datetime.utcnow())
b_utt_3 = BotUtterance(text='в 1672 году', user=b_user, active_skill='odqa', confidence=0.74,
                       date_time=datetime.utcnow())

h_utt_4 = Utterance(text='спасибо', user=h_user, date_time=datetime.utcnow())

utterances = [h_utt_1, b_utt_1, h_utt_2, b_utt_2, h_utt_3, b_utt_3, h_utt_4]
d = Dialog(utterances=utterances, users=[h_user, BOT], channel_type='telegram')

h_user.save()
b_user.save()

h_utt_1.save()
b_utt_1.save()

h_utt_2.save()
b_utt_2.save()

h_utt_3.save()
b_utt_3.save()

h_utt_4.save()

d.save()

h_user_2 = Human(user_telegram_id=uuid.uuid4())
h_utt_5 = Utterance(text='Когда началась Вторая Мировая?', user=h_user, date_time=datetime.utcnow())
b_utt_5 = BotUtterance(text='1939', user=b_user, active_skill='odqa', confidence=0.99,
                       date_time=datetime.utcnow())
h_utt_6 = Utterance(text='Спасибо, бот!', user=h_user, date_time=datetime.utcnow())
utterances_1 = [h_utt_5, b_utt_5, h_utt_6]
d_1 = Dialog(utterances=utterances_1, users=[h_user_2, BOT], channel_type='telegram')
h_user_2.save()
h_utt_5.save()
b_utt_5.save()
h_utt_6.save()
d_1.save()
#
# count = 0
# total = {'version': 0.9}
#
# batch = []
# for d in Dialog.objects:
#     if count < 2:
#         info = d.to_dict()
#         batch.append(info)
#         count += 1

for d in Dialog.objects:
    print(d.to_dict())


