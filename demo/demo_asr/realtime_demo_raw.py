import asyncio
import time

import sounddevice as sd
import pyaudio
import numpy as np
from scipy.io import wavfile
from custom_wave import write_binary
from zijie.release_interface import get_client
appid = "6747655566"    # 项目的 appid
token = "M_3Swzuc6aTtP90HE6VHQ58NmBdF_6Rl"    # 项目的 token
cluster = "volcengine_streaming_common"  # 请求的集群
audio_format = "raw"   # wav 或者 mp3，根据实际音频格式设置
channel = 2
asr_client = get_client(
        {
            'id': 1
        },
        cluster=cluster,
        appid=appid,
        token=token,
        format=audio_format,
        show_utterances=True,
        channel=channel
)
fs = 16000 # sample rate
seconds = 10


count = 0
while True:
    print('start record...')
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=channel, dtype=np.int16)
    sd.wait()
    wave_binrary = myrecording.tobytes()
    count += 1
    # result = asyncio.run(asr_client.execute(f'./{count-1}.wav', True))
    print('ending...record, \n start send')
    s = time.time()
    result = asyncio.run(asr_client.execute_raw(wave_binrary, 1, 16, fs))
    e = time.time()
    print(f'end send, cost: {e - s}')
    if result['payload_msg']['message'] == 'Success':
        print(result['payload_msg']['result'][0]['text'])
    else:
        print(result)
    if count >= 100:
        break

# bytestream = tone_out.tobytes()
# pya = pyaudio.PyAudio()
# stream = pya.open(format=pya.get_format_from_width(width=2), channels=1,
#                   rate=OUTPUT_SAMPLE_RATE, output=True)
# stream.write(bytestream)
# stream.stop_stream()
# stream.close()
#
# pya.terminate()
# print("* Preview completed!")