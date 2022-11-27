import time

from src.commu.client import SocketClient
from src.language.question import Questions
import json


class GlobalStatus(object):
    def __init__(self, ip, port):
        self.current_question_id = -1
        self.stop_in_listen = False
        self.asr_in_listen = False
        self.is_stop = False
        self.send_msg_client = SocketClient()
        self.send_msg_client.connet(ip, port)

        self.questions = Questions()
        # self.send_msg_client = None


def audio_receive_message(conn, audio_processor_obj, global_status:GlobalStatus):
    while True:
        messages = conn.recv(1024).decode('utf-8')
        # messages = json.loads(messages)
        # content = messages.get('order', None)
        content = messages
        print(f'receive: {content}')
        if content is None:
            print(f'ignore message: {messages}')
            continue
        if content == 'StartProgram':
            print('start interview')
            # audio_processor_obj.audio_stop_object.reset()
            # audio_processor_obj.audio_asr_object.reset()

            global_status.send_msg_client.send_asr_txt('虚拟人：' + global_status.questions.questions[0].content)

            time.sleep(0.1)

            global_status.send_msg_client.send_message(json.dumps({
                'order': '0'
            }))
            global_status.current_question_id = 0
        elif content == 'AudioFinish':
            # 播放完成，开始监听说话是否停止，并且开始读取ASR的相关信息
            # 开启监听
            global_status.asr_in_listen = True
            global_status.stop_in_listen = True
            global_status.is_stop = False
