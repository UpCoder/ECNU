import sounddevice as sd
import numpy as np
from zijie.release_interface import get_client
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
                 n_channels, stop_interval=1, stop_threshold=3000):
        self.sample_rate = sample_rate
        self.record_duration = record_duration
        self.n_channels = n_channels
        self.records = []
        self.stop_interval = stop_interval
        self.stop_threshold = stop_threshold

    def demo_realtime(self):
        while True:
            print('start rec')
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
                print('安静了~')

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


if __name__ == '__main__':
    audit_processor = AudioStopRecord(sample_rate=fs, record_duration=seconds, n_channels=channel,
                                  stop_interval=3, stop_threshold=3000)
    audit_processor.demo_realtime()
