#coding:utf-8
import time
import os
from src.commu.client import SocketClient
from src.language.question import Questions
import json

global_json_path = '../../环境变量.json'
if os.path.exists(global_json_path):
    config_json_obj = json.load(open(global_json_path, 'r', encoding='utf-8'))
else:
    config_json_obj = dict()


class GlobalStatus(object):
    def __init__(self, ip, port):
        self.current_question_id = -1
        self.stop_in_listen = False
        self.asr_in_listen = False
        self.is_stop = False
        self.send_msg_client = SocketClient()
        self.send_msg_client.connet(ip, port)

        self.questions = Questions()
        self.sample_rate = config_json_obj.get('语音-采样频率', 16000)
        self.asr_record_duration = config_json_obj.get('语音-翻译间隔', 5)
        self.stop_threshold = config_json_obj.get('语音-安静阈值', 6000)
        # self.send_msg_client = None

    def reset(self):
        self.current_question_id = -1
        self.stop_in_listen = False
        self.asr_in_listen = False
        self.is_stop = False


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
                'order': '0',
                'speech_tone': str(0),
                'speech_pause_count': str(0),
                'speech_loudness': '{:.5f}'.format(0),
                'speech_speed': str(0),
                'speech_pitch': str(0),
                'speech_length': str(0)
            }))
            global_status.current_question_id = 0
        elif content == 'AudioFinish':
            # 播放完成，开始监听说话是否停止，并且开始读取ASR的相关信息
            # 开启监听
            global_status.asr_in_listen = True
            global_status.stop_in_listen = True
            global_status.is_stop = False
        elif content == 'StopProgram':
            global_status.reset()
            audio_processor_obj.reset()


if __name__ == '__main__':
    print(config_json_obj)
