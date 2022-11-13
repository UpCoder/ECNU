import moviepy.editor as mpy
import wave
from io import BytesIO
from moviepy.editor import VideoFileClip, TextClip

mp4_path = 'C:\\Users\\cs_li\\Documents\\WXWork\\1688854406374298\\Cache\\Video\\2022-10\\3.mp4'

audit_background = mpy.AudioFileClip(mp4_path)
audit_background.write_audiofile('3.wav')


def read_wav_info(data: bytes = None) -> (int, int, int, int, int):
    with BytesIO(data) as _f:
        wave_fp = wave.open(_f, 'rb')
        nchannels, sampwidth, framerate, nframes = wave_fp.getparams()[:4]
        wave_bytes = wave_fp.readframes(nframes)
    return nchannels, sampwidth, framerate, nframes, len(wave_bytes)

def demo_wav_info():
    print(read_wav_info(
        bytes(open('../tts.wav', 'rb').read())
    ))


def writr_zimu(video_path, zimu_infos, save_path):
    video = VideoFileClip(video_path)
    txt_clips = []
    for zimu_info in zimu_infos:
        txt_clips.append(
            TextClip(
                zimu_info['content'],
                fontsize=70,
                color='black'
            ).with_position('center').with_
        )


if __name__ == '__main__':
    demo_wav_info()