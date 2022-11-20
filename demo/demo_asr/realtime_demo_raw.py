import argparse
import asyncio
import threading
import time
import sounddevice as sd
import numpy as np
from zijie.release_interface import get_client
from src.commu.client import SocketClient


class AudioASRRecord(object):
    def __init__(self, sample_rate=16000, record_duration=5,
                 n_channels=2, args=None, client=None):
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
        self.process_idx = 0
        self.asr_result_thread = None
        self.realtime_two_thread = None
        if client is None:
            self.socket_client = SocketClient()
            self.socket_client.connet(
                host=args.host,
                port=args.port
            )
        else:
            self.socket_client = client


    def start_get_asr_result_thread(self):
        if self.asr_result_thread is None or not self.asr_result_thread.is_alive():
            self.asr_result_thread = threading.Thread(target=self.get_asr_result_thread)
            self.asr_result_thread.setDaemon(True)
            self.asr_result_thread.start()

    def get_asr_result_thread(self):
        while True:
            if len(self.records) == self.process_idx:
                time.sleep(0.1)
                continue
            cur_record = self.records[self.process_idx]
            self.process_idx += 1
            asr_result = self.get_asr_result(cur_record.tobytes())
            print(f'asr_result:{asr_result}')
            # TODO send msg
            self.socket_client.send_asr_txt(asr_result)
            continue

    def get_asr_result(self, record_bytes):
        result = asyncio.run(self.asr_client.execute_raw(record_bytes, self.channel,
                                                         self.bits, self.fs))
        print(f'result: {result}')
        if result['payload_msg']['message'] == 'Success':
            return result['payload_msg']['result'][0]['text']
        return ''

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
            print('finish record...')
            # wave_binrary = myrecording.tobytes()
            self.records.append(myrecording)
            count += 1
            if count >= 100:
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demo')
    parser.add_argument('--host', type=str, default="192.168.0.107")
    parser.add_argument('--port', type=int, default=8889)
    args = parser.parse_args()
    print(args)
    print('init object')
    audit_asr_processor = AudioASRRecord(args=args, record_duration=5)
    print('finish init object')
    # audit_asr_processor.demo_realtime()
    print('start get asr result')
    audit_asr_processor.start_get_asr_result_thread()
    print('start loop')
    # audit_asr_processor.demo_realtime_two_thread()
    audit_asr_processor.start_demo_realtime_two_thread()
    time.sleep(1000000)