import asyncio

import sounddevice as sd
import pyaudio
import numpy as np
from scipy.io import wavfile
from custom_wave import write_binary
from zijie.release_interface import get_client
appid = "6747655566"    # 项目的 appid
token = "M_3Swzuc6aTtP90HE6VHQ58NmBdF_6Rl"    # 项目的 token
cluster = "volcengine_streaming_common"  # 请求的集群
audio_format = "wav"   # wav 或者 mp3，根据实际音频格式设置
asr_client = get_client(
        {
            'id': 1
        },
        cluster=cluster,
        appid=appid,
        token=token,
        format=audio_format,
        show_utterances=True
)
fs = 16000 # sample rate
seconds = 15


count = 0
while True:
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype=np.int16)
    print(type(myrecording))
    sd.wait()
    # wavfile.write(f'./{count}.wav', fs, myrecording)
    wave_binrary = write_binary(fs, myrecording)
    count += 1
    # result = asyncio.run(asr_client.execute(f'./{count-1}.wav', True))
    result = asyncio.run(asr_client.execute(wave_binrary, False))
    print(result)
    if result['payload_msg']['message'] == 'Success':
        print(result['payload_msg']['result'][0]['text'])
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