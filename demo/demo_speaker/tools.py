import os
from glob import glob
from pydub import AudioSegment


def mp42wav(mp4_path, wav_path, nchannel, rate):
    command = f'ffmpeg -i {mp4_path} -ac {nchannel} -ar {rate} {wav_path}'
    print(command)
    status = os.system(command)
    print(f'exec result: {status}')


def change_channels(wav_path: str):
    if wav_path.endswith('.wav'):
        sound = AudioSegment.from_wav(wav_path)
        sound = sound.set_channels(1)
        sound.export(wav_path, format='wav')
    else:
        print(f'DO NOT SUPPORT! {wav_path}')


if __name__ == '__main__':
    mp4_dir = 'C:\\Users\\cs_li\\Downloads\\TALK_VIDEO'
    for mp4_path in glob(os.path.join(mp4_dir, '*.mp4')):
        print(f'handler: {mp4_path}')
        wav_path = mp4_path.replace('.mp4', '.wav')
        if os.path.exists(wav_path):
            continue
        mp42wav(
            mp4_path,
            wav_path,
            1,
            16000
        )