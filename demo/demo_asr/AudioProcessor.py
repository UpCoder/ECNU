import socket
import threading
import json
import logging
import time

from src.commu.client import SocketClient
from realtime_demo_raw import AudioASRRecord
from realtime_if_stop import AudioStopRecord
from utils import GlobalStatus, audio_receive_message


global global_status


class AudioProcessor(object):
    def __init__(self, sample_rate, stop_record_duration, asr_record_duration,
                 n_channels, stop_interval=1, stop_threshold=3000,
                 global_status:GlobalStatus=None):
        self.sample_rate = sample_rate
        self.stop_record_duration = stop_record_duration
        self.asr_record_duration = asr_record_duration
        self.n_channels = n_channels
        self.stop_interval = stop_interval
        self.stop_threshold = stop_threshold
        self.global_status = global_status

        self.audio_asr_object = AudioASRRecord(sample_rate, record_duration=asr_record_duration,
                                               n_channels=n_channels, global_status=global_status)
        self.audio_stop_object = AudioStopRecord(sample_rate, stop_record_duration, n_channels,
                                                 stop_interval, stop_threshold, global_status=global_status)
        # 开启监听
        # self.audio_stop_object.start_listen_is_stop_thread()
        self.audio_asr_object.start_get_asr_result_thread()
        self.audio_asr_object.start_realtime_recording_thread()

    def reset(self):
        self.audio_asr_object = AudioASRRecord(self.sample_rate, record_duration=self.asr_record_duration,
                                               n_channels=self.n_channels, global_status=self.global_status)
        self.audio_stop_object = AudioStopRecord(self.sample_rate, self.stop_record_duration, self. n_channels,
                                                 self.stop_interval, self.stop_threshold,
                                                 global_status=self.global_status)
        # 开启监听
        # self.audio_stop_object.start_listen_is_stop_thread()
        self.audio_asr_object.start_get_asr_result_thread()
        self.audio_asr_object.start_realtime_recording_thread()

def start_pipeline(ip, port):
    global global_status
    # sock = socket.socket()
    # print('start bind')
    # sock.bind((ip, port))
    # print('listen')
    # sock.listen(1)
    # while True:
    #     try:
            # 接收来自服务器的数据
            # conn, addr = sock.accept()
            # global_status = GlobalStatus('localhost', 8889)
            # audio_processor = AudioProcessor(
            #     sample_rate=16000, asr_record_duration=5,
            #     stop_record_duration=1, n_channels=2,
            #     stop_interval=3, stop_threshold=3000,
            #     global_status=global_status
            # )
            # print('receive after.')
            # thread1 = threading.Thread(target=audio_receive_message, args=(global_status.send_msg_client.socket_client,
            #                                                                audio_processor,
            #                                                                global_status,))
            # thread1.start()
        # except Exception as e:
        #     logging.exception(e)
        #     break
    global_status = GlobalStatus('localhost', 8889)
    audio_processor = AudioProcessor(
        sample_rate=16000, asr_record_duration=5,
        stop_record_duration=1, n_channels=2,
        stop_interval=3, stop_threshold=3000,
        global_status=global_status
    )
    print('receive after.')
    thread1 = threading.Thread(target=audio_receive_message, args=(global_status.send_msg_client.socket_client,
                                                                   audio_processor,
                                                                   global_status,))
    thread1.start()


if __name__ == '__main__':
    start_pipeline('localhost', 8880)
    time.sleep(1000000)