#coding:utf-8
import logging
import time
import os
from src.commu.client import SocketClient
from src.language.question import Questions, SeqQuestions
import json
from threading import Lock
import pyaudio

global_json_path = 'E:\\PycharmProjects\\ECNU\\环境变量.json'
if os.path.exists(global_json_path):
    config_json_obj = json.load(open(global_json_path, 'r', encoding='utf-8'))
else:
    config_json_obj = dict()


class GlobalStatus(object):
    def __init__(self, ip, port):
        audio_device = AudioDevice()

        self.current_question_id = -1
        self.stop_in_listen = False
        self.asr_in_listen = False
        self.is_stop = False
        self.is_stop_lock = Lock()
        self.send_msg_client = SocketClient()
        self.send_msg_client.connet(ip, port)

        # self.questions = Questions()
        # self.questions = SeqQuestions()
        self.sample_rate = config_json_obj.get('语音-采样频率', 16000)
        self.asr_record_duration = config_json_obj.get('语音-翻译间隔', 5)
        self.stop_threshold = config_json_obj.get('语音-安静阈值', 6000)
        self.stop_sec_threshold = config_json_obj.get('语音-安静时长（单位秒）', 2)
        self.think_sec_threshold = config_json_obj.get('语音-问题思考时间（单位秒）', 5)
        self.audio_channel = audio_device.num_channel
        self.is_generated_audio_data = config_json_obj.get('生成语音', False)
        if self.is_generated_audio_data:
            self.questions = SeqQuestions()
        else:
            self.questions = Questions()
        # self.send_msg_client = None

    def reset(self):
        self.current_question_id = -1
        self.stop_in_listen = False
        self.asr_in_listen = False
        self.is_stop = False


def audio_receive_message(conn, audio_processor_obj, global_status: GlobalStatus):
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

            # global_status.send_msg_client.send_asr_txt('虚拟人：' + global_status.questions.questions[0].content)

            time.sleep(0.1)
            if global_status.is_generated_audio_data:
                audio_data, audio_duration_s = global_status.questions.generate_audio_data(
                    global_status.questions.questions[0].content)
            else:
                audio_data = ''
                audio_duration_s = 0
            logging.info(f'data: {audio_data}')
            global_status.send_msg_client.send_message(json.dumps({
                'dialogue': '虚拟人：' + global_status.questions.questions[0].content,
                'order': '0',
                'have_audio_data': '1' if global_status.is_generated_audio_data else '0',
                'length': str(audio_duration_s),
                'data': audio_data,
                'speech_tone': str(0),
                'speech_pause_count': str(0),
                'speech_loudness': '{:.5f}'.format(0),
                'speech_speed': str(0),
                'speech_pitch': str(0),
                'speech_length': str(0),
                'word_num': '0',
                'sentence_num': '0',
                'pos_word_num': '0',
                'neg_word_num': '0',
                'n': '0',
                'r': '0',
                'v': '0',
                'a': '0',
                'good_word': '0',
                'happy_word': '0',
                'sad_word': '0',
                'angry_word': '0',
                'fear_word': '0',
                'disgust_word': '0',
                'surprise_word': '0'
            }))
            global_status.current_question_id = 0
        elif content == 'AudioFinish':
            # 播放完成，开始监听说话是否停止，并且开始读取ASR的相关信息
            # 开启监听
            global_status.is_stop_lock.acquire()
            try:
                global_status.asr_in_listen = True
                global_status.stop_in_listen = True
                global_status.is_stop = False
            except Exception as e:
                print(f'AudioFinish occur exception: {e}')
            finally:
                global_status.is_stop_lock.release()
        elif content == 'EndRecord':
            # 结束录音
            global_status.is_stop_lock.acquire()
            try:
                print('end record start')
                global_status.is_stop = True
                print(f'end record end, {global_status.is_stop} {id(global_status)}')
            except Exception as e:
                print(f'EndRecord occur exception: {e}')
            finally:
                global_status.is_stop_lock.release()
        elif content == 'StopProgram':
            global_status.reset()
            audio_processor_obj.reset()


class AudioDevice(object):
    def __init__(self):
        obj = pyaudio.PyAudio().get_default_input_device_info()
        self.num_channel = obj['maxInputChannels']
        print(self.num_channel)


if __name__ == '__main__':
    # print(config_json_obj)
    device = AudioDevice()
