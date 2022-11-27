import time

import sounddevice as sd
import numpy as np
import threading
from zijie.release_interface import get_client
from utils import GlobalStatus
appid = "6747655566"    # 项目的 appid
token = "M_3Swzuc6aTtP90HE6VHQ58NmBdF_6Rl"    # 项目的 token
cluster = "volcengine_streaming_common"  # 请求的集群
audio_format = "raw"   # wav 或者 mp3，根据实际音频格式设置
channel = 2
bits = 16
asr_client = get_client(
        {
            'id': 1
        },
        cluster=cluster,
        appid=appid,
        token=token,
        format=audio_format,
        show_utterances=True,
        channel=channel,
        bits=bits
)
fs = 16000  # sample rate
seconds = 1  # 单位是秒


class AudioStopRecord(object):
    def __init__(self, sample_rate, record_duration,
                 n_channels, stop_interval=1, stop_threshold=3000, global_status:GlobalStatus=None):
        self.sample_rate = sample_rate
        self.record_duration = record_duration
        self.n_channels = n_channels
        self.records = []
        self.stop_interval = stop_interval
        self.stop_threshold = stop_threshold
        self.is_quiet = False   # 初始化的状态是不安静的，监听到这个状态变化之后再去发送消息
        self.speech_is_stop_thread = None
        self.global_status = global_status

    def realtime_core(self):
        while True:
            if not self.global_status.stop_in_listen:
                # 如果没有监听的话就不会录音
                time.sleep(0.001)
                continue
            myrecording = sd.rec(int(self.record_duration * self.sample_rate),
                                 samplerate=self.sample_rate, channels=self.n_channels, dtype=np.int16)
            sd.wait()
            max_record = np.max(myrecording)
            min_record = np.min(myrecording)
            print(f'end rec max_record: {type(myrecording)} {max_record} {min_record}')
            self.records.append({
                'record': myrecording,
                'max': max_record,
                'min': min_record
            })
            if self.is_stop():
                self.global_status.is_stop = True
                print('安静了~')
            else:
                # self.is_quiet = False
                self.global_status.is_stop = False

    def is_stop(self):
        batch_size = self.stop_interval // self.record_duration
        if len(self.records) < batch_size:
            return False
        latest_info = self.records[len(self.records) - batch_size:]
        maxs = np.asarray([info['max'] for info in latest_info])
        maxs_bool = np.asarray(maxs < self.stop_threshold, np.bool)
        if np.all(maxs_bool):
            return True
        return False

    def start_listen_is_stop_thread(self):
        if self.speech_is_stop_thread is None or not self.speech_is_stop_thread.is_alive():
            self.speech_is_stop_thread = threading.Thread(target=self.realtime_core)
            self.speech_is_stop_thread.setDaemon(True)
            self.speech_is_stop_thread.start()


if __name__ == '__main__':
    audit_processor = AudioStopRecord(sample_rate=fs, record_duration=seconds, n_channels=channel,
                                      stop_interval=3, stop_threshold=3000)
    audit_processor.start_listen_is_stop_thread()
    time.sleep(1000000)
