import logging
import os
from glob import glob
import argparse
from demo.demo_speaker.demo_speaker import handler_wav2sentence, handler_sentence2asr
from demo.demo_speaker.tools import change_channels, mp42wav
from src.audio.asr_tools.file_handler import handler_pipeline


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demo')
    parser.add_argument('--video_dir', type=str, default='E:\\videos0328')
    parser.add_argument('--spk_id_huoshan', type=bool, default=True)
    args = parser.parse_args()
    mp4_dir = args.video_dir
    txt_dir = os.path.join(mp4_dir, 'txt')
    if not os.path.exists(txt_dir):
        os.makedirs(txt_dir)
    for mp4_path in glob(os.path.join(mp4_dir, '*.mp4')):
        try:
            print(f'handler: {mp4_path}')
            wav_path = mp4_path.replace('.mp4', '.wav')
            if os.path.exists(wav_path):
                os.remove(wav_path)
            mp42wav(
                mp4_path,
                wav_path,
                1,
                16000
            )
            change_channels(wav_path)
            if not args.spk_id_huoshan:
                txt_path = handler_wav2sentence(wav_path, txt_dir)
                handler_sentence2asr(txt_path, wav_path)
            else:
                txt_path = handler_pipeline(wav_path, txt_dir)
                handler_sentence2asr(txt_path, wav_path, False, 'json')
        except Exception as e:
            logging.exception(e)