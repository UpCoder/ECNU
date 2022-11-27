import argparse
import asyncio
import json
import logging
import random
import threading
import multiprocessing
import time
import sounddevice as sd
import numpy as np
from zijie.release_interface import get_client
from src.commu.client import SocketClient
from utils import GlobalStatus


class AudioASRRecord(object):
    def __init__(self, sample_rate=16000, record_duration=5,
                 n_channels=2, args=None, client=None,
                 stop_interval=5, stop_threshold=3000,
                 global_status:GlobalStatus=None):
        self.stop_interval = stop_interval
        self.stop_threshold = stop_threshold
        self.appid = "6747655566"    # 项目的 appid
        self.token = "M_3Swzuc6aTtP90HE6VHQ58NmBdF_6Rl"    # 项目的 token
        self.cluster = "volcengine_streaming_common"  # 请求的集群
        self.audio_format = "raw"   # wav 或者 mp3，根据实际音频格式设置
        self.channel = n_channels
        self.bits = 16
        self.asr_client = get_client(
                {
                    'id': 1
                },
                cluster=self.cluster,
                appid=self.appid,
                token=self.token,
                format=self.audio_format,
                show_utterances=True,
                channel=self.channel,
                bits=self.bits
        )
        # sample rate
        self.fs = sample_rate
        self.seconds = record_duration
        self.records = []
        self.asrs = []
        self.process_idx = 0
        self.asr_result_thread = None
        self.realtime_two_thread = None
        self.demo_asr_result_thread = None
        self.start_realtime_recording_thread_obj = None
        self.global_status = global_status
        self.stop_records = []
        self.question_answers = []
        self.cur_question_answer = ''
        self.answer_seconds = []   # 记录历史问题回答的秒数
        self.cur_seconds = 0    # 记录当前问题回答的秒数
        if client is None:
            self.socket_client = self.global_status.send_msg_client
        else:
            self.socket_client = client
        self.in_listen = False

    def start_get_asr_result_thread(self):
        if self.asr_result_thread is None or not self.asr_result_thread.is_alive():
            self.asr_result_thread = threading.Thread(target=self.get_asr_result_thread)
            self.asr_result_thread.setDaemon(True)
            self.asr_result_thread.start()

    def get_asr_result_thread(self):
        while True:
            if not self.global_status.asr_in_listen:
                time.sleep(0.001)
                continue
            if len(self.records) == self.process_idx:
                time.sleep(0.1)
                continue
            if self.process_idx >= len(self.records):
                continue
            cur_record = self.records[self.process_idx]
            self.process_idx += 1
            asr_result = self.get_asr_result(cur_record.tobytes())
            print(f'asr_result: {asr_result}')
            print(asr_result == '')
            self.asrs.append(asr_result)
            self.cur_question_answer += asr_result
            # TODO send msg
            if asr_result != '':
                self.socket_client.send_asr_txt('被试：' + asr_result)
            if self.global_status.is_stop:
                # send next play audio
                self.global_status.asr_in_listen = False
                self.global_status.is_stop = False
                self.question_answers.append(self.cur_question_answer)
                print('next quest')
                next_question_id = self.global_status.questions.get_next_question(
                    self.global_status.current_question_id, self.cur_question_answer)
                self.cur_question_answer = ''

                self.answer_seconds.append(self.cur_seconds)
                self.cur_seconds = 0

                self.global_status.current_question_id = next_question_id
                # self.global_status.current_question_id += 1

                self.global_status.send_msg_client.send_asr_txt('虚拟人：' + self.global_status.questions.questions[
                    self.global_status.current_question_id].content)
                time.sleep(0.1)
                self.global_status.send_msg_client.send_message(json.dumps(
                    {"order": str(self.global_status.current_question_id)}
                ))
                time.sleep(0.1)
                self.global_status.send_msg_client.send_message(json.dumps(
                    self.calc_metrics()
                ))
            continue

    def calc_metrics(self):
        """
        计算相关指标
        :return:
        """
        # 计算响度
        batch_size = self.answer_seconds[-1] // self.seconds
        data = self.stop_records[len(self.stop_records) - batch_size:]
        loudness = []
        for single_record in data:
            # print(single_record[np.where(single_record['record'] >= 0)])
            loudness.append(np.mean(single_record[single_record['record']]))

        # 计算语速
        speech_speed = len(self.question_answers[-1]) / (self.answer_seconds[-1] * 60)

        # 计算说话时长
        speech_length = self.answer_seconds[-1]

        # 计算音调变化
        speech_tone = ''

        # 计算停顿次数
        speech_pause_count = 0

        # 计算高音pitch
        speech_pitch = 0

        return {
            'speech_tone': str(speech_tone),
            'speech_pause_count': str(speech_pause_count),
            'speech_loudness': '{:.5f}'.format(np.mean(loudness)),
            'speech_speed': str(speech_speed),
            'speech_pitch': str(speech_pitch),
            'speech_length': str(speech_length)
        }



    def get_asr_result(self, record_bytes):
        result = asyncio.run(self.asr_client.execute_raw(record_bytes, self.channel,
                                                         self.bits, self.fs))
        print(f'result: {result}')
        if result['payload_msg']['message'] == 'Success':
            return result['payload_msg']['result'][0]['text']
        return ''

    def start_realtime_recording_thread(self):
        if self.start_realtime_recording_thread_obj is None or not self.start_realtime_recording_thread_obj.is_alive():
            self.start_realtime_recording_thread_obj = threading.Thread(target=self.realtime_recording_thread_core)
            self.start_realtime_recording_thread_obj.setDaemon(True)
            self.start_realtime_recording_thread_obj.start()

    def realtime_recording_thread_core(self):
        count = 0
        while True:
            if not self.global_status.asr_in_listen:
                time.sleep(0.001)
                continue
            if self.global_status.is_stop:
                time.sleep(0.001)
                continue
            print('start record...')
            myrecording = sd.rec(int(self.seconds * self.fs), samplerate=self.fs,
                                 channels=self.channel, dtype=np.int16)
            sd.wait()
            max_record = np.max(myrecording)
            min_record = np.min(myrecording)
            self.cur_seconds += self.seconds
            print(f'end rec max_record: {type(myrecording)} {np.shape(myrecording)} {max_record} {min_record}')
            self.stop_records.append({
                'record': myrecording,
                'max': max_record,
                'min': min_record
            })
            print('finish record...')
            # wave_binrary = myrecording.tobytes()
            self.records.append(myrecording)
            if self.is_stop():
                self.global_status.is_stop = True
                print('安静了~')
            else:
                # self.is_quiet = False
                self.global_status.is_stop = False
            count += 1
            if count >= 100:
                break

    def demo_realtime(self):
        count = 0
        while True:
            print('start record...')
            myrecording = sd.rec(int(self.seconds * self.fs), samplerate=self.fs,
                                 channels=self.channel, dtype=np.int16)
            print(type(myrecording), np.max(myrecording))
            sd.wait()
            print('finish', np.max(myrecording), np.min(myrecording))
            wave_binrary = myrecording.tobytes()
            count += 1
            print('ending...record, \n start send')
            s = time.time()
            result = asyncio.run(self.asr_client.execute_raw(wave_binrary, self.channel,
                                                             self.bits, self.fs))
            e = time.time()

            print(f'end send, cost: {e - s}')
            if result['payload_msg']['message'] == 'Success':
                print(result['payload_msg']['result'][0]['text'])
            else:
                print(result)
            if count >= 100:
                break

    def start_demo_realtime_two_thread(self):
        if self.realtime_two_thread is None or not self.realtime_two_thread.is_alive():
            self.realtime_two_thread = threading.Thread(target=self.demo_realtime_two_thread)
            self.realtime_two_thread.setDaemon(True)
            self.realtime_two_thread.start()


    def demo_realtime_two_thread(self):
        count = 0
        while True:
            print('start record...')
            myrecording = sd.rec(int(self.seconds * self.fs), samplerate=self.fs,
                                 channels=self.channel, dtype=np.int16)
            sd.wait()
            max_record = np.max(myrecording)
            min_record = np.min(myrecording)
            print(f'end rec max_record: {type(myrecording)} {max_record} {min_record}')
            self.stop_records.append({
                'record': myrecording,
                'max': max_record,
                'min': min_record
            })
            print('finish record...')
            # wave_binrary = myrecording.tobytes()
            self.records.append(myrecording)
            if self.is_stop():
                print('安静了~')
            else:
                # self.is_quiet = False
                pass
            count += 1
            if count >= 100:
                break

    def start_demo_get_asr_result_thread(self):
        if self.demo_asr_result_thread is None or not self.demo_asr_result_thread.is_alive():
            self.demo_asr_result_thread = threading.Thread(target=self.demo_get_asr_result_thread)
            self.demo_asr_result_thread.setDaemon(True)
            self.demo_asr_result_thread.start()

    def demo_get_asr_result_thread(self):
        while True:
            if self.process_idx >= len(self.records):
                continue
            cur_record = self.records[self.process_idx]
            self.process_idx += 1
            asr_result = self.get_asr_result(cur_record.tobytes())
            print(f'asr_result: {asr_result}')
            print(asr_result == '')
            self.asrs.append(asr_result)
            self.cur_question_answer += asr_result
            continue

    def is_stop(self):
        batch_size = self.stop_interval // self.seconds
        if len(self.stop_records) < batch_size:
            return False
        latest_info = self.stop_records[len(self.stop_records) - batch_size:]
        maxs = np.asarray([info['max'] for info in latest_info])
        maxs_bool = np.asarray(maxs < self.stop_threshold, np.bool)
        if np.all(maxs_bool):
            return True
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demo')
    parser.add_argument('--host', type=str, default="192.168.0.107")
    parser.add_argument('--port', type=int, default=8889)
    args = parser.parse_args()
    print(args)
    print('init object')
    audit_asr_processor = AudioASRRecord(args=args,
                                         record_duration=5,
                                         client='debug')
    print('finish init object')
    # audit_asr_processor.demo_realtime()
    print('start get asr result')
    audit_asr_processor.start_demo_get_asr_result_thread()
    print('start loop')
    # audit_asr_processor.demo_realtime_two_thread()
    audit_asr_processor.start_demo_realtime_two_thread()
    time.sleep(1000000)