import pyaudio
import time
import threading
import wave
import winsound
import matplotlib.pyplot as pl
import numpy as np
import sys
from PyQt5.Qt import *


class Recorder:
    def __init__(self, chunk=1024, channels=2, rate=64000):
        self.CHUNK = chunk
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = channels
        self.RATE = rate
        self._running = True
        self._frames = []

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
            data = stream.read(self.CHUNK)
            self._frames.append(data)
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


# 创建应用程序
app = QApplication(sys.argv)
re = Recorder()

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

# lable.show()
window.show()

# 等待窗口停止
sys.exit(app.exec())