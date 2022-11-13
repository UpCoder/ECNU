import cython
print(cython.__version__)
import pyworld as pw
from ttskit import sdk_api

# wav = sdk_api.tts_sdk('文本', audio='24')
# print(type(wav))
with open('./tts.txt', 'r') as f:
    lines = f.readlines()
lines = ''.join(lines)
wav = sdk_api.tts_sdk(lines, audio='24')
with open('tts.wav', 'wb') as f:
    f.write(wav)
# with open('./test.wav', 'wb') as f:
#    f.write(wav)