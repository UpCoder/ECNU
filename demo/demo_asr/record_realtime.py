import queue

import pyaudio
import time
import threading
import wave
import winsound
import matplotlib.pyplot as pl
import numpy as np
import math
import sys
import asyncio
import json
from demo.demo_asr.zijie.release_interface import get_client
from demo.demo_asr.utils import GlobalStatus
from PyQt5.Qt import *
from src.language.verbal import VerbalAnalyzer


def merge_asr(cur_asr_result: str, last_asr_result: str):
    """
    合并两句话
    :param cur_asr_result:
    :param last_asr_result:
    :return:
    """
    location_length = 5    # 置信的长度
    end_position = len(last_asr_result)
    while (end_position - location_length) > 0:
        find_str = last_asr_result[end_position-location_length: end_position]
        end_position -= 1
        idx = cur_asr_result.find(find_str)
        if idx == -1:
            continue
        return last_asr_result[:end_position] + cur_asr_result[idx+len(find_str)-1:]
    return cur_asr_result


class QAData(object):
    def __init__(self):
        """
        存放QA过程中需要存储/使用的数据
        """
        self.frames_qa_format = [[]]  # 二维数组，里面是一个list，存放每次answer的list
        self.merged_asr_results_qa_format = [[]]  # 二维数组，里面是一个list，存放每次answer更新的ASR结果
        self.current_asr_results_qa_format = [[]]
        self.current_answer_results = []
        self.current_merged_results = []
        self.processed_chunk_idx = 0


class AudioRecorderProcessor:
    def __init__(self, chunk=16000, channels=2,
                 rate=16000, max_asr_window=30,
                 asr_interval_sec=1,
                 stop_sec_threshold=3,
                 stop_value_threshold=6000,
                 think_sec_threshold=5,
                 global_status: GlobalStatus = None,
                 recorder_queue: queue.Queue = None):
        self.CHUNK = chunk
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = channels
        self.RATE = rate
        self._running = True
        self._frames = []   # 一维数组的格式
        self.sec_per_chunk = chunk / rate   # 每个chunk代表的实际录音时长
        self.appid = "6747655566"  # 项目的 appid
        self.token = "M_3Swzuc6aTtP90HE6VHQ58NmBdF_6Rl"  # 项目的 token
        self.cluster = "volcengine_streaming_common"  # 请求的集群
        self.audio_format = "raw"  # wav 或者 mp3，根据实际音频格式设置
        self.channel = channels
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
        self.asr_result_thread_obj = None
        self.start_realtime_recording_thread_obj = None
        self.asr_interval_sec = asr_interval_sec   # asr 提交的间隔，单位s
        self.origin_asr_results = []   # 每次ASR识别的结果，原始结果
        self.merged_asr_results = []   # 每次ASR识别的结果，合并处理过
        self.print_asr_loop_flag = False
        self.max_asr_window = max_asr_window    # ASR最大的窗口，每次最多识别30s的数据，如果小于30s，则用实际的长度，如果大于30s，则用最新的30s

        self.stop_sec_threshold = stop_sec_threshold
        self.stop_value_threshold = stop_value_threshold
        self.think_sec_threshold = think_sec_threshold  # 每个问题开始的等待时间
        self.listen_is_stop_flag = False

        self.global_status = global_status
        self.recorder_queue = recorder_queue
        self.qa_data = QAData()

        self.verbal_analyzer = VerbalAnalyzer()

    def add_recorder_msg(self, data_dict: dict = None):
        """
        留存日志信息
        :return:
        """
        if self.recorder_queue is not None and data_dict is not None:
            self.recorder_queue.put({
                'type': 'dict',
                'origin': 'audio',
                **data_dict
            })

    def calc_metrics_stop_interval(self, record, stop_duration_threshold_ms):
        """
        计算停顿的次数，和每次停顿的时长
        :param record
        :param stop_duration_threshold_ms:
        :return:
        """

        cur_record = np.asarray(np.abs(record), np.int16)
        batch_size = self.RATE // 1000  # 代表每ms的数据量
        is_stop = []
        for i in range(int(np.shape(cur_record)[0] // batch_size)):
            # print(i * batch_size, (i+1) * batch_size)
            cur_slice = cur_record[i * batch_size: (i+1) * batch_size, :]
            # print(i * batch_size, (i + 1) * batch_size, np.max(cur_slice), np.min(cur_slice))
            if np.max(cur_slice) <= self.stop_value_threshold:
                is_stop.append(1)
            else:
                is_stop.append(0)
        cur_count_stop = 0
        record_stop_durations_ms = []
        for cur_is_stop in is_stop:
            if cur_is_stop == 0:
                record_stop_durations_ms.append(cur_count_stop)
                cur_count_stop = 0
            elif cur_is_stop == 1:
                cur_count_stop += 1
        if cur_count_stop != 0:
            record_stop_durations_ms.append(cur_count_stop)
        record_stop_durations_ms = list(filter(lambda x: x > stop_duration_threshold_ms,
                                               record_stop_durations_ms))
        return len(record_stop_durations_ms), record_stop_durations_ms

    def calc_metrics(self, default=False):
        """
        计算相关指标
        :return:
        """
        if not default:
            # 时间
            cost_secs = len(self.qa_data.frames_qa_format[-1]) * self.sec_per_chunk
            # 计算响度
            data = [np.reshape(np.frombuffer(binary_value, dtype=np.int16), [-1, 2])
                    for binary_value in self.qa_data.frames_qa_format[-1]]
            # data = [self.stop_records[-1]]
            loudness = []
            # 计算高音pitch
            speech_pitch = -np.inf
            speech_pause_count = 0
            speech_pause_duration_ms = []
            for single_record in data:
                # print(single_record[np.where(single_record['record'] >= 0)])
                loudness.append(np.mean(single_record))
                speech_pitch = max(speech_pitch, np.max(single_record))
                # 计算停顿次数
                speech_pause_count_, speech_pause_duration_ms_ = self.calc_metrics_stop_interval(
                    single_record, 300)
                speech_pause_count += speech_pause_count_
                speech_pause_duration_ms.extend(speech_pause_duration_ms_)
            # 计算语速
            speech_speed = len(self.qa_data.current_merged_results[-1]) / cost_secs

            # 计算说话时长
            speech_length = cost_secs

            # 计算音调变化
            speech_tone = ''

            # verbal
            verbal_metrics = self.verbal_analyzer.run(self.qa_data.current_merged_results[-1])

            metric_dict = {
                'speech_tone': str(speech_tone),    # 音调变化
                # 'speech_pause': str(speech_pause_count),  # 停顿次数
                'speech_loudness': '{:.4f}'.format(np.mean(loudness)),  # 平均响度
                'speech_speed': '{:.4f}'.format(speech_speed),  # 语速
                'speech_pitch': '{:.4f}'.format(speech_pitch),  # 高音pitch
                'speech_length': '{:.4f}'.format(speech_length),    # 回答问题的时长
                # 'speech_pause_duration': '{:.5f}'.format(np.sum(speech_pause_duration_ms))   # 停顿的总时长
                **verbal_metrics
            }
        else:
            verbal_metrics = self.verbal_analyzer.run('')
            metric_dict = {
                'speech_tone': '',  # 音调变化
                'speech_loudness': '',  # 平均响度
                'speech_speed': '',
                'speech_pitch': '',
                'speech_length': '',
                **verbal_metrics
            }
        return metric_dict

    def start_realtime_recording_thread(self):
        if self.start_realtime_recording_thread_obj is None or not self.start_realtime_recording_thread_obj.is_alive():
            self.start_realtime_recording_thread_obj = threading.Thread(target=self.realtime_recording_core)
            self.start_realtime_recording_thread_obj.setDaemon(True)
            self.start_realtime_recording_thread_obj.start()

    def realtime_recording_core(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=self.FORMAT,
                        channels=self.CHANNELS,
                        rate=self.RATE,
                        input=True,
                        frames_per_buffer=self.CHUNK)
        while True:
            self.global_status.is_stop_lock.acquire()
            try:
                # print(f'is_stop1: {self.global_status.is_stop} {id(self.global_status)}')
                if not self.global_status.asr_in_listen:
                    time.sleep(0.001)
                    continue
                if self.global_status.is_stop:
                    time.sleep(0.001)
                    continue
            finally:
                self.global_status.is_stop_lock.release()
            print('start record...')
            s = time.time()
            data = stream.read(self.CHUNK)
            e = time.time()
            self._frames.append(data)
            self.qa_data.frames_qa_format[-1].append(data)
            self.global_status.is_stop_lock.acquire()
            try:
                if self.is_stop(self.qa_data.frames_qa_format[-1]):
                    self.global_status.is_stop = True
                else:
                    # self.global_status.is_stop = False
                    # 注释上面这行，避免覆盖receive endrecord
                    pass
            except Exception as e:
                print(f'Record Is Stop Error {e}')
            finally:
                self.global_status.is_stop_lock.release()
    def start(self):
        threading._start_new_thread(self.__recording, ())

    def __recording(self):
        self._running = True
        print('start recording')
        self._frames = []
        p = pyaudio.PyAudio()
        stream = p.open(format=self.FORMAT,
                        channels=self.CHANNELS,
                        rate=self.RATE,
                        input=True,
                        frames_per_buffer=self.CHUNK)

        while (self._running):
            s = time.time()
            data = stream.read(self.CHUNK)
            e = time.time()
            self._frames.append(data)
            self.qa_data.frames_qa_format[-1].append(data)
        stream.stop_stream()
        stream.close()
        p.terminate()

        # def stop(self):
        self._running = False

    def save(self):
        self._running = False

        p = pyaudio.PyAudio()

        wf = wave.open("001.wav", 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(p.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(self._frames))
        wf.close()
        print("Saved")

    def play(self):
        winsound.PlaySound("001.wav", winsound.SND_FILENAME)
        # 打开wav文档
        file = wave.open("001.wav", "rb")

        # 读取参数信息
        # nchannels, sampwidth, framerate, nframes, comptype, compname
        params = file.getparams()
        nchannels, sampwidth, framerate, nframes = params[:4]
        print(params)

        # 将字符转格式的数据转成int型
        str_data = file.readframes(nframes)
        wave_data = np.frombuffer(str_data, dtype=np.short)

        # 归一化
        wave_data = wave_data * 1.0 / max(abs(wave_data))

        # 将音频信号规整成每行一路通道信号的格式，即该矩阵一行为一个通道的采样点，共nchannels行
        wave_data = np.reshape(wave_data, [nframes, nchannels]).T  # T表示转置
        print(wave_data)

        # 文件使用完毕，关闭文件
        file.close()

        # 绘制语音波形
        time = np.arange(0, nframes) * (1.0 / framerate)  # 时间=n/fs
        time = np.reshape(time, [nframes, 1]).T
        pl.plot(time[0, :nframes], wave_data[0, :nframes], c="b")
        pl.xlabel("time(seconds)")
        pl.ylabel("amplitude")
        pl.title("original wave")
        pl.show()

    def get_asr_result_core(self, frames=None):
        max_frames = self.max_asr_window // self.sec_per_chunk
        if frames is None:
            binary_data = b''.join(self._frames[int(-1 * max_frames):])
        else:
            binary_data = b''.join(frames[int(-1 * max_frames):])
        arrive_limit = max_frames <= len(self._frames)
        result = asyncio.run(self.asr_client.execute_raw(binary_data, self.channel,
                                                         self.bits, self.RATE))
        if result['payload_msg']['message'] == 'Success':
            cur_asr_result = result['payload_msg']['result'][0]['text']
            if not arrive_limit:
                # 如果没有到达限制，则不需要和前一个进行合并
                merged_asr_result = cur_asr_result
            else:
                # 如果到达了限制，则需要和前一个ASR的结果进行合并
                merged_asr_result = merge_asr(cur_asr_result, self.merged_asr_results[-1])
                # self.merged_asr_results.append(merged_asr_result)
            # print(f'merged asr result: {self.merged_asr_results[-1]}')
            return cur_asr_result, merged_asr_result
        else:
            return '', ''

    def print_asr_loop(self):
        """
        获取asr的result，每次都是从头获取
        :return:
        """
        while self.print_asr_loop_flag:
            cur_asr_result, merged_asr_result = self.get_asr_result_core()
            self.merged_asr_results.append(merged_asr_result)
            self.origin_asr_results.append(cur_asr_result)
            time.sleep(self.asr_interval_sec)

    def get_asr_result_thread(self):
        while True:
            self.global_status.is_stop_lock.acquire()
            cur_loop_stop = None
            try:
                # print(f'asr result is stop 1: {self.global_status.is_stop} {id(self.global_status)}')
                if not self.global_status.asr_in_listen:
                    time.sleep(0.001)
                    continue
                if self.qa_data.processed_chunk_idx >= len(self.qa_data.frames_qa_format[-1]):
                    time.sleep(0.001)
                    continue
                cur_loop_stop = self.global_status.is_stop
            except Exception as e:
                print(f'ASR result Error: {e}')
            finally:
                self.global_status.is_stop_lock.release()
            cur_asr_result, merged_asr_result = self.get_asr_result_core(self.qa_data.frames_qa_format[-1])
            self.qa_data.processed_chunk_idx += 1
            # print(f'cur_asr_result: {cur_asr_result}\nmerged_asr_result: {merged_asr_result}')
            self.qa_data.current_answer_results.append(cur_asr_result)
            self.qa_data.current_merged_results.append(merged_asr_result)
            self.qa_data.merged_asr_results_qa_format[-1].append(merged_asr_result)
            self.qa_data.current_asr_results_qa_format[-1].append(cur_asr_result)
            self.origin_asr_results.append(cur_asr_result)
            self.merged_asr_results.append(merged_asr_result)
            # self.answers[-1].cur_answer = merged_asr_result
            # TODO send msg
            if merged_asr_result != '':
                self.add_recorder_msg(
                    {
                        'asr': '被试：' + merged_asr_result
                    }
                )
                self.global_status.send_msg_client.send_message(
                    json.dumps(
                        {
                            'dialogue': '被试：' + merged_asr_result,
                            **self.calc_metrics(True)
                        }
                    )
                )
            # print(f'asr result is stop 2: {self.global_status.is_stop}')
            if cur_loop_stop:
                '''
                判断停止了才会进行下述操作
                1. 更新状态
                2. 根据本轮回答结果，计算下一轮的问题ID
                3. 计算相关指标
                4. 留存相关数据
                5. 重置相关状态
                '''
                # self.question_answers.append(self.cur_question_answer)

                print('next quest')
                next_question_id = self.global_status.questions.get_next_question(
                    self.global_status.current_question_id, self.qa_data.current_merged_results[-1])

                self.global_status.current_question_id = next_question_id
                if self.global_status.is_generated_audio_data:
                    audio_data, audio_duration_s = self.global_status.questions.generate_audio_data(
                        self.global_status.questions.questions[next_question_id].content)
                else:
                    audio_data = ''
                    audio_duration_s = 0
                time.sleep(0.1)
                self.global_status.send_msg_client.send_message(json.dumps(
                    {
                        'dialogue': '虚拟人：' + self.global_status.questions.questions[
                            self.global_status.current_question_id].content,
                        "order": str(self.global_status.current_question_id),
                        "have_audio_data": '1' if self.global_status.is_generated_audio_data else '0',
                        'length': str(audio_duration_s),
                        'data': audio_data,
                        **self.calc_metrics()
                    },
                ))
                self.add_recorder_msg(
                    {
                        'asr': '虚拟人：' + self.global_status.questions.questions[
                            self.global_status.current_question_id].content
                    }
                )
                self.add_recorder_msg({
                    **self.calc_metrics()
                })

                self.add_recorder_msg({
                    'action': 'save_wav',
                    'data': np.reshape(np.frombuffer(b''.join(self.qa_data.frames_qa_format[-1]),
                                       dtype=np.int16), [-1, 2]),
                    'fs': self.RATE
                })

                # Re-Init
                self.qa_data.frames_qa_format.append([])
                self.qa_data.current_asr_results_qa_format.append([])
                self.qa_data.merged_asr_results_qa_format.append([])
                self.qa_data.current_merged_results = []
                self.qa_data.current_answer_results = []
                self.qa_data.processed_chunk_idx = 0
                self.global_status.is_stop_lock.acquire()
                try:
                    self.global_status.asr_in_listen = False
                    self.global_status.is_stop = False
                except Exception as e:
                    print(f'asr set stop: {self.global_status.is_stop}')
                finally:
                    self.global_status.is_stop_lock.release()
            continue

    def start_get_asr_result_thread(self):
        if self.asr_result_thread_obj is None or not self.asr_result_thread_obj.is_alive():
            self.asr_result_thread_obj = threading.Thread(target=self.get_asr_result_thread)
            self.asr_result_thread_obj.setDaemon(True)
            self.asr_result_thread_obj.start()

    def start_print_asr_loop(self):
        self.print_asr_loop_flag = True
        self.print_asr_loop()

    def end_print_asr_loop(self):
        self.print_asr_loop_flag = False

    def is_stop(self, frames=None):
        num_chunks = math.ceil(self.stop_sec_threshold / self.sec_per_chunk)
        think_num_chunks = math.ceil(self.think_sec_threshold / self.sec_per_chunk)
        print(f'num_chunks: {num_chunks} / {len(frames) if frames is not None else len(self._frames)},'
              f'think_num_chunks: {think_num_chunks}')
        if frames is None:
            binary_values = self._frames[int((-1 * num_chunks)):]
        else:
            binary_values = frames[int((-1 * num_chunks)):]
        if len(binary_values) < num_chunks or len(frames) < think_num_chunks:
            return False
        binary_values = b''.join(binary_values)
        np_values = np.frombuffer(binary_values, dtype=np.int16)
        if np.all(np_values < self.stop_value_threshold):
            print(f'安静了, {np.max(np_values)}')
            return True
        else:
            print(f'max_value: {np.max(np_values)}')
            return False

    def print_is_stop(self):
        """
        判断是否停止
        :return:
        """
        while self.listen_is_stop_flag:
            self.is_stop()
            time.sleep(1)

    def start_print_is_stop(self):
        self.listen_is_stop_flag = True
        self.print_is_stop()


    def start_qt(self):
        """
        测试一下TQT的UI
        :return:
        """
        # 创建应用程序
        app = QApplication(sys.argv)
        rate = 16000    # 每秒采样的frame个数
        channels = 2
        chunk = rate    # Specifies the number of frames per buffer. chunk=rate代表每秒存储一次，chunk=rate*0.5,则代表500毫秒采样一次
        re = AudioRecorderProcessor(rate=rate, channels=channels, chunk=chunk)

        # 创建窗口
        window = QWidget()
        window.setWindowTitle("111")
        window.resize(500, 500)

        btn1 = QPushButton(window)
        btn1.move(100, 50)
        btn1.setText("播放并画图")
        btn1.clicked.connect(re.play)

        btn2 = QPushButton(window)
        btn2.move(100, 90)
        btn2.setText("开始录音")
        btn2.clicked.connect(re.start)

        btn3 = QPushButton(window)
        btn3.move(100, 130)
        btn3.setText("结束录音")
        btn3.clicked.connect(re.save)


        btn4 = QPushButton(window)
        btn4.move(100, 170)
        btn4.setText("开启ASR")
        btn4.clicked.connect(re.start_print_asr_loop)


        btn5 = QPushButton(window)
        btn5.move(100, 210)
        btn5.setText("结束ASR")
        btn5.clicked.connect(re.end_print_asr_loop)

        btn6 = QPushButton(window)
        btn6.move(100, 250)
        btn6.setText("开始判断是否结束说话")
        btn6.clicked.connect(re.start_print_is_stop)

        # lable.show()
        window.show()

        # 等待窗口停止
        sys.exit(app.exec())

    @staticmethod
    def start_test_pipeline():
        global_status = GlobalStatus('localhost', 8889)
        num_channels = 1
        recorder = AudioRecorderProcessor(max_asr_window=30, global_status=global_status, channels=num_channels)
        global_status.asr_in_listen = True
        global_status.stop_in_listen = True
        global_status.is_stop = False
        recorder.start_realtime_recording_thread()
        recorder.start_get_asr_result_thread()
        time.sleep(1000000)


if __name__ == '__main__':
    # audit_recorder = AudioRecorderProcessor(
    #     max_asr_window=30
    # )
    # audit_recorder.start_qt()
    AudioRecorderProcessor.start_test_pipeline()