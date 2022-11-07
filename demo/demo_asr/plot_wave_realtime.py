import wave
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
from matplotlib.animation import FuncAnimation
import pyaudio
import time
import threading
from scipy.fftpack import fft, ifft,irfft
from threading import Lock,Thread
import codecs
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['font.sans-serif'] = ['SimHei']   #显示中文
mpl.rcParams['axes.unicode_minus']=False       #显示负号

class Audiowave:

    def __init__(self):

        self.wavedata=[]
        self.wavewidth=2
        self.wavechannel=1
        self.framerate = 48000
        self.fps=24
        self.Timedata=[]
        self.nframes=0
        self.N=0
        self.data=[]
        self.dataall = []
        np.array(self.dataall)
        # plt.figure(figsize=(16, 8))
        self.fig, self.ax = plt.subplots(2,1,figsize=(8, 6))
        self.p=pyaudio.PyAudio()  # 实例化对象
        self.q=pyaudio.PyAudio()  # 实例化对象
        if self.wavewidth == 1:
            format= pyaudio.paInt8
        elif self.wavewidth == 2:
            format= pyaudio.paInt16
        elif self.wavewidth == 3:
            format= pyaudio.paInt24
        elif self.wavewidth == 4:
            format= pyaudio.paFloat32
        # fps=waveframerate/waveCHUNK  #每秒数据更新次数
        self.waveCHUNK = int(self.framerate / self.fps)
        # a=time.time()
        self.stream =self.p.open(format=format,
                        channels=self.wavechannel,
                        rate=self.framerate,
                        input=True,
                        frames_per_buffer=self.waveCHUNK)  #录音
        self.stream1 =self.q.open(format=format,
                        channels=self.wavechannel,
                        rate=self.framerate,
                        output=True,
                        frames_per_buffer=self.waveCHUNK) #播放
        self.c = 0
        self.a = time.time()
        self.b = 0
        self.x=[]
        self.y=[]
        self.fft_int=[]
        np.array(self.x)
        np.array(self.y)
        self.wf = wave.open(r"./test.wav", 'wb')
        self.wf.setnchannels(self.wavechannel)  # 声道设置
        self.wf.setsampwidth(self.p.get_sample_size(format))  # 采样位数设置
        self.wf.setframerate(self.framerate)
        self.wf1 = wave.open(r"./test1.wav", 'wb')
        self.wf1.setnchannels(self.wavechannel)  # 声道设置
        self.wf1.setsampwidth(self.p.get_sample_size(format))  # 采样位数设置
        self.wf1.setframerate(self.framerate)

    def waveopen(self, filedir):
        wf = wave.open(filedir, "rb")
        self.nframes = wf.getnframes()
        self.wavedata = wf.readframes(self.nframes)
        self.wavewidth=wf.getsampwidth()
        self.wavechannel=wf.getnchannels()
        self.framerate = wf.getframerate()
        self.time=self.nframes/self.framerate
        bps=self.framerate*self.wavewidth*8*self.wavechannel
        print("总帧数："+str(self.nframes)+"帧")
        print("采样率："+str(self.framerate)+"帧/s")
        print("声道数："+str(self.wavechannel)+"个")
        print("位深："+str(self.wavewidth*8)+"bit")
        print("比特率："+str(bps/1000)+"kbps")
        print("时间："+ str(self.time)+"s")
        print("文件大小："+ str(self.time*bps/8/1024/1024)+"MB")
        return self.wavedata,self.wavewidth,self.wavechannel,self.framerate,self.time



    def wavehex_to_DEC(self,wavedata,wavewidth,wavechannel):
        print("#####################")


        print(type(self.Timedata))
        n = int(len(wavedata) / wavewidth)
        i = 0
        j = 0
        for i in range(0, n):
            b = 0
            for j in range(0, wavewidth):
                temp = wavedata[i * wavewidth:(i + 1) * wavewidth][j] * int(math.pow(2, 8 * j))
                b += temp
            if b > int(math.pow(2, 8 * wavewidth - 1)):
                b = b - int(math.pow(2, 8 * wavewidth))
            self.Timedata.append(b)
        self.Timedata = np.array(self.Timedata)
        # print(len(self.Timedata))
        self.Timedata.shape = -1, wavechannel
        self.Timedata = self.Timedata.T

        x = np.linspace(0, len(self.Timedata[0])-1, len(self.Timedata[0])) / self.framerate

        return x,self.Timedata

    def to_fft(self,N,data):#转换为频域数据，并进行取半、归一化等处理
        # N=self.nframes #取样点数
        df = self.framerate / (N - 1)  #每个点分割的频率 如果你采样频率是4096，你FFT以后频谱是从-2048到+2048hz（4096除以2），然后你的1024个点均匀分布，相当于4个HZ你有一个点，那个复数的模就对应了频谱上的高度
        freq = [df * n for n in range(0, N)]

        wave_data2 = data[0:N]
        self.fft_int=np.fft.fft(wave_data2)
        # print(N, len(data),len(wave_data2))
        c = self.fft_int * 2 / N  #*2能量集中化  /N归一化
        d = int(len(c) / 2)  #对称取半
        freq=freq[:d - 1]
        fredata=abs(c[:d - 1])

        return freq,fredata



    def wavedraw1(self,filedir):
        self.waveopen(filedir)
        timedata=self.wavehex_to_DEC(self.wavedata,self.wavewidth,self.wavechannel)
        fredata=self.to_fft(self.nframes,timedata[1][0])


        plt.figure(figsize=(16, 8))
        plt.subplot(2,1,1)
        plt.plot(timedata[0], timedata[1][0])
        plt.xlabel(u"Time(S)")
        plt.subplot(212)
        plt.plot(fredata[0],fredata[1])
        plt.xlim(0, 800)
        plt.xlabel(u"Freq(Hz)")

        plt.subplots_adjust(hspace=0.4)
        plt.show()

    def wavedrawall(self,filedir):
        self.waveopen(filedir)
        n=self.wavechannel
        timedata = self.wavehex_to_DEC(self.wavedata, self.wavewidth, self.wavechannel)
        xtimedata=timedata[0]
        ytimedata=timedata[1]
        # print(ytimedata)
        i=0
        plt.figure(figsize=(16, 8))
        # plt.legend(prop=font_set)
        for i in range(0,n):
            a=plt.subplot(2, n, i+1)
            a.set_title("track"+str(i+1))
            plt.plot(xtimedata, ytimedata[i])
            plt.xlabel(u"Time(S)")
            # print(ytimedata[i])
            fredata = self.to_fft(self.nframes,ytimedata[i])
            plt.subplot(2,n,i+1+n)
            plt.plot(fredata[0], fredata[1])
            plt.xlim(0, 4000)
            plt.xlabel(u"Freq(Hz)")
        plt.subplots_adjust(hspace=0.4)
        plt.show()

    def Dynamic_init(self):

        # self.fig.figsize(16,8)
        # print(self.Timedata)
        # Y_max=max(list(self.Timedata)
        Y_max=max(self.Timedata)*5/4
        self.ax[0].set_xlim(0, int(self.time)*5/4)
        self.ax[0].set_ylim(-(Y_max),(Y_max))
        self.ax[1].set_xlim(0, 10000)
        # self.ax.set_ylim(0,20000)
        ln, = self.ax[0].plot([], [], animated=False)
        return ln,  # 返回曲线

    def Dynamic_updata(self,n):
        # print("#####",n)
        N=self.N
        xdata=np.linspace(0,len(self.Timedata),len(self.Timedata)+1)
        # xdata.pop()
        xdata=np.delete(xdata,-1)
        ydata=self.Timedata
        # print(len(xdata),len(ydata))
        # print(ydata)
        x_t=xdata[0:int((n+1)*N)]/self.framerate
        # print(len(x_t))
        y_t=ydata[0:int((n+1)*N)]
        fre_data=self.Timedata[int(n*N):int((n+1)*N)]
        if len(fre_data)>0:
            a=self.to_fft(N,fre_data)
            x=a[0]
            y=a[1]
        else:
            x=0
            y=0
        # print(x,y)

        ln,=self.ax[0].plot(x_t,y_t,"g-")
        ln1,=self.ax[1].plot(x,y,"g-")

        # ln1=ln# 重新设置曲线的值
        return ln,ln1,

    def Dynamic_run(self,filedir): #
        self.waveopen(filedir)
        FPS=24
        self.N=int(self.framerate/FPS)#每次fft取样点数
        self.Timedata = self.wavehex_to_DEC(self.wavedata, self.wavewidth, self.wavechannel)[1][0]
        print(len(self.Timedata))
        temp = list(np.linspace(0, int(self.nframes/self.N),int(self.nframes/self.N)+1))
        # temp.pop()
        ani = FuncAnimation(self.fig, self.Dynamic_updata, frames=temp, interval=self.N/self.framerate*1000,
                            init_func=self.Dynamic_init, blit=True)
        plt.show()

    def wavehex_to_DEC_n(self,wavedata,wavewidth,wavechannel):#录音存储数据十六进制数据转换十进制
        # print("#####################")
        Timedata=[]


        # print(type(self.Timedata))
        n = int(len(wavedata) / wavewidth)
        i = 0
        j = 0
        for i in range(0, n):
            b = 0
            for j in range(0, wavewidth):
                temp = wavedata[i * wavewidth:(i + 1) * wavewidth][j] * int(math.pow(2, 8 * j))
                b += temp
            if b > int(math.pow(2, 8 * wavewidth - 1)):
                b = b - int(math.pow(2, 8 * wavewidth))
            Timedata.append(b)
        Timedata = np.array(Timedata)
        # print(len(self.Timedata))
        Timedata.shape = -1, wavechannel
        Timedata = Timedata.T

        x = np.linspace(0, len(Timedata[0])-1, len(Timedata[0])) / self.framerate

        return x,Timedata


    def DEC_to_wavehex(self,DEC_data, wavewidth=2, wavechannel=1):
        wavewidth=self.wavewidth
        wavedata = ""
        for data in DEC_data:
            data = int(data)

            if data < 0:
                data += int(math.pow(2, 8 * wavewidth))

                # print(data)
            a = hex(data)[2:]
            a = a[::-1]
            while len(a) < 2 * wavewidth:
                a += "0"
            for i in range(0, wavewidth):
                # print(a[i * 2:2 * i + 2])
                b = r"\x" + a[i * 2:2 * i + 2]
                wavedata += b
            # wavedata.append(b)
        wavedata=bytes(wavedata, encoding="utf8")

        return codecs.escape_decode(wavedata, "hex-escape")[0]


    def micdata(self):#录音数据，存储，播放
        t00 = time.time()
        data = self.stream.read(self.waveCHUNK) #录音
        # data=b'\x00\x00\x00\x00\xff\xff'
        print(data)
        print(len(data),type(data))

        self.data = data
        filter_data = self.filter(data, 0, 600)   ###这个要转成16进制进行播放
        # print(filter_data)
        # print(filter_data)
        # print(len(filter_data))

        # print(wave_data)
        t11=time.time()
        # print(t11-t00)
        # #
        # print(len(data))
        # print(len(filter_data))
        # print(data==filter_data)
        # x=np.linspace(0,len(filter_data),len(filter_data))
        # plt.subplot(2,1,1)
        # plt.plot(x,data)
        # plt.subplot(2,1,2)
        # plt.plot(x,filter_data)
        # print(len(wave_data),type(wave_data))
        # print(wave_data)
        # print("###########################")
        # print(data)
        # print(wave_data)
        self.stream1.write(data) #播放
        self.wf.writeframes(data)#写入存储
        self.wf1.writeframes(filter_data)#写入存储
        return data, filter_data



        # print(len(self.dataall))
        # b=time.time()
        # print(b-a)
        # print(data)
        return data

    def Dynamic_micwave_init(self):#动态显示图像-初始化图像

        self.ax[0].set_xlim(0, 1/self.fps)
        self.ax[1].set_xlim(0,2000)

        ln, = self.ax[0].plot([], [], animated=False)
        return ln,  # 返回曲线

    def Dynamic_micwave_update(self,n):#动态显示图像-更新图像

        # print(self.data)
        # b=time.time()
        # d=b-self.a
        # self.a=b
        # self.c+=d
        # print(self.c)
        x, y = self.wavehex_to_DEC_n(self.data, self.wavewidth, self.wavechannel)

        # self.dataall=np.append(self.dataall,y[0])
        # self.b+=len(y[0])
        # x_t = np.linspace(0, self.c, self.b)
        # y_t = self.dataall
        # # print(len(x_t),len(y_t))
        # # print(len(self.dataall))
        # if self.c>10:
        #     print(self.c)
        #     self.ax[0].set_xlim(self.c-10, self.c)
        # # print(len(self.dataall))
        fre_x,fre_y=self.to_fft(self.waveCHUNK,y[0])

        ln, = self.ax[0].plot(x, y[0], "g-")
        # print("###############")
        # print(len(fre_y))
        ln1, = self.ax[1].plot(fre_x, fre_y, "g-")

        return ln,ln1,

    def Dynamic_micwave_run(self): #动态显示图像

        ani = FuncAnimation(self.fig,
                            self.Dynamic_micwave_update,
                            interval=1000/self.fps,
                            init_func=self.Dynamic_micwave_init,
                            blit=True)
        plt.show()

    def filter(self, data, m, n):
        x,DEC_data=self.wavehex_to_DEC_n(data,self.wavewidth,self.wavechannel)
        self.to_fft(self.waveCHUNK, DEC_data[0])
        # print(DEC_data[0][0:10])
        # print(fre_x[-1])
        # print(DEC_data)
        a=int(m*(self.waveCHUNK-1)/self.framerate)
        b=int(n*(self.waveCHUNK-1)/self.framerate)

        # print("#####")
        # print(len(self.fft_int))
        # print(a,b)
        # print(self.fft_int[0:10])
        self.fft_int[a:b]=0
        self.fft_int[-b:-a]=0

        # print(self.fft_int[0:10])
        # print(self.fft_int)
        ifft_y = np.fft.ifft(self.fft_int).real
        # print(ifft_y[0:10])
        ifft_y=np.trunc(ifft_y)
        wave_data = self.DEC_to_wavehex(ifft_y)
        return wave_data

a=Audiowave()


def recorder_run():
    while True:
        a.micdata()


def draw_run():
    a.Dynamic_micwave_run()


if __name__ == '__main__':
    t1 = threading.Thread(target=recorder_run, args=())     # target是要执行的函数名（不是函数），args是函数对应的参数，以元组的形式存在
    # t2 = threading.Thread(target=draw_run,args=())
    # t2.setDaemon(True)
    t1.setDaemon(True)
    t1.start()
    a.Dynamic_micwave_run()
    time.sleep(100000)