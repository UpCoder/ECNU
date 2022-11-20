import matplotlib.pyplot as plt
import wave
import numpy as np


def plow_wave_by_wav_file(wav_file):
    f = wave.open(wav_file, 'rb')
    # nchannels, sampwidth, framerate, nframes, comptype, compname
    paras = f.getparams()
    nchannels, sample_width, frame_rate, n_frames = paras[:4]
    print(f'通道个数:{nchannels}\n 采样率:{frame_rate}\n 采样点数:{n_frames}\n sample_width: {sample_width}')
    str_data = f.readframes(n_frames)
    f.close()
    print(len(str_data))
    wave_data = np.fromstring(str_data, dtype=np.short)
    wave_data = wave_data * 1.0 / (max(abs(wave_data)))
    print(len(wave_data))
    # 整合左声道和右声道的数据
    wave_data = np.reshape(wave_data, [n_frames, nchannels])
    # wave_data.shape = (-1, 2)   # -1的意思就是没有指定,根据另一个维度的数量进行分割

    # 最后通过采样点数和取样频率计算出每个取样的时间
    time = np.arange(0, n_frames) * (1.0 / frame_rate)

    plt.figure()
    # 左声道波形
    plt.subplot(2, 1, 1)
    plt.plot(time, wave_data[:, 0])
    plt.xlabel("时间/s", fontsize=14)
    plt.ylabel("幅度", fontsize=14)
    plt.title("左声道", fontsize=14)
    plt.grid()  # 标尺

    plt.subplot(2, 1, 2)
    # 右声道波形
    # plt.plot(time, wave_data[:, 1], c="g")
    # plt.xlabel("时间/s", fontsize=14)
    # plt.ylabel("幅度", fontsize=14)
    # plt.title("右声道", fontsize=14)

    plt.tight_layout()  # 紧密布局
    plt.show()


if __name__ == '__main__':
    plow_wave_by_wav_file('./0.wav')