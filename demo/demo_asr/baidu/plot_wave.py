import numpy as np  # 科学计算所用的numpy库
import matplotlib.pyplot as plt  # 绘图所用的库matplotlib
from scipy.io import wavfile  # 读取波形文件所用的库

rate_h, hstrain = wavfile.read('1.wav', 'rb')

rate_l, lstrain = wavfile.read("1.wav", "rb")
# # reftime, ref_H1 = np.genfromtxt('GW150914_4_NR_waveform_template.txt').transpose()
# reftime, ref_H1 = np.genfromtxt('wf_template.txt').transpose()  # 使用python123.io下载txt文件

htime_interval = 1 / rate_h
ltime_interval = 1 / rate_l
fig = plt.figure(figsize=(12, 6))  # 创建大小为12*6的绘图空间

# 丢失信号起始点
htime_len = hstrain.shape[0] / rate_h  # 读取数据第一维的长度，得到函数在坐标轴上总长度
htime = np.arange(-htime_len / 2, htime_len / 2, htime_interval)  # （起点，终点，时间间隔）

plth = fig.add_subplot(221)  # 设置绘图区域
plth.plot(htime, hstrain, 'r')  # 画出以时间为x轴，应变数据为y轴的图像，‘y'为黄色
plth.set_xlabel('Time (seconds)')
plth.set_ylabel('H1 Strain')
plth.set_title('H1 Strain')
plt.show()