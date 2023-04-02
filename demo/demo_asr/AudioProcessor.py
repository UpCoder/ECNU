import queue
import socket
import threading
import json
import logging
import time

from demo.demo_asr.record_realtime import AudioRecorderProcessor
from demo.demo_asr.utils import GlobalStatus, audio_receive_message


global global_status


class AudioProcessor(object):
    def __init__(self, sample_rate, stop_record_duration, asr_record_duration,
                 n_channels, stop_interval=1, stop_threshold=3000,
                 global_status: GlobalStatus = None,
                 recorder_queue: queue.Queue = None):
        self.sample_rate = sample_rate
        self.stop_record_duration = stop_record_duration
        self.asr_record_duration = asr_record_duration
        self.n_channels = n_channels
        self.stop_interval = stop_interval
        self.stop_threshold = stop_threshold
        self.global_status = global_status
        self.recorder_queue = recorder_queue
        # self.audio_recorder_processor_object = AudioASRRecord(sample_rate, record_duration=asr_record_duration,
        #                                        n_channels=n_channels, global_status=global_status,
        #                                        recorder_queue=self.recorder_queue)
        self.audio_recorder_processor_object = AudioRecorderProcessor(
            rate=sample_rate, channels=n_channels, global_status=global_status,
            recorder_queue=recorder_queue, stop_sec_threshold=global_status.stop_sec_threshold,
            stop_value_threshold=global_status.stop_threshold,
            think_sec_threshold=global_status.think_sec_threshold
        )
        # self.audio_stop_object = AudioStopRecord(sample_rate, stop_record_duration, n_channels,
        #                                          stop_interval, stop_threshold, global_status=global_status)
        # 开启监听
        # self.audio_stop_object.start_listen_is_stop_thread()
        self.audio_recorder_processor_object.start_get_asr_result_thread()
        self.audio_recorder_processor_object.start_realtime_recording_thread()

    def reset(self):
        # self.audio_asr_object = AudioASRRecord(self.sample_rate, record_duration=self.asr_record_duration,
        #                                        n_channels=self.n_channels,
        #                                        global_status=self.global_status,
        #                                        recorder_queue=self.recorder_queue)
        # self.audio_stop_object = AudioStopRecord(self.sample_rate, self.stop_record_duration, self. n_channels,
        #                                          self.stop_interval, self.stop_threshold,
        #                                          global_status=self.global_status)
        # 开启监听
        # self.audio_asr_object.start_get_asr_result_thread()
        # self.audio_asr_object.start_realtime_recording_thread()
        self.audio_recorder_processor_object = AudioRecorderProcessor(
            rate=self.sample_rate, channels=self.n_channels, global_status=global_status,
            recorder_queue=self.recorder_queue, stop_sec_threshold=global_status.stop_sec_threshold,
            stop_value_threshold=global_status.stop_threshold,
            think_sec_threshold=global_status.think_sec_threshold
        )
        self.audio_recorder_processor_object.start_get_asr_result_thread()
        self.audio_recorder_processor_object.start_realtime_recording_thread()


def start_pipeline(recorder_queue: queue.Queue = None):
    global global_status
    global_status = GlobalStatus('localhost', 8889)
    audio_processor = AudioProcessor(
        sample_rate=global_status.sample_rate,
        asr_record_duration=global_status.asr_record_duration,
        stop_record_duration=1, n_channels=global_status.audio_channel,
        stop_interval=3, stop_threshold=global_status.stop_threshold,
        global_status=global_status, recorder_queue=recorder_queue
    )
    print('receive after.')
    thread1 = threading.Thread(target=audio_receive_message, args=(global_status.send_msg_client.socket_client,
                                                                   audio_processor,
                                                                   global_status,))
    thread1.start()


if __name__ == '__main__':
    from src.commu.record import Recorder
    import os
    timestamp_now = time.time()
    record_file_path = os.path.join('./', '{}_record.txt'.format(timestamp_now))
    wav_dir = os.path.join('./wav', '{}'.format(timestamp_now))
    recorder = Recorder(record_file_path, wav_dir)
    recorder.start()
    start_pipeline(recorder.q1)
    time.sleep(1000000)