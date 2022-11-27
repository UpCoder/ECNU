import os
from ttskit import sdk_api
print('finish import')


def demo():
    wav = sdk_api.tts_sdk('文本', audio='24')
    with open('tts.wav', 'wb') as f:
        f.write(wav)


def general_wavs(file_path, output_dir=None):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for idx, line in enumerate(lines):
        wav_bin = sdk_api.tts_sdk(line, audio='24')
        if output_dir is None:
            output_dir = './'
        output_path = os.path.join(output_dir, f'{idx}.wav')
        with open(output_path, 'wb') as f:
            f.write(wav_bin)


if __name__ == '__main__':
    general_wavs(
        # 'C:\\Users\\cs_li\\Documents\\大五人格访谈视频+简短问卷\\1120访谈流程.txt',
        # 'C:\\Users\\cs_li\\Documents\\大五人格访谈视频+简短问卷\\访谈流程Audios'
        'C:\\Users\\cs_li\\PycharmProjects\\body_seg\\src\\language\\questions1.txt',
        'C:\\Users\\cs_li\\PycharmProjects\\body_seg\\src\\language\\question1Audios',
    )