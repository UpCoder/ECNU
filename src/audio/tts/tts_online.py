import base64
import json
import logging
import time
import uuid
import requests
import io
from pydub import AudioSegment
from pydub.playback import play


# 填写平台申请的appid, access_token以及cluster
appid = "6747655566"    # 项目的 appid
token = "M_3Swzuc6aTtP90HE6VHQ58NmBdF_6Rl"    # 项目的 token
cluster = "volcano_tts"  # 请求的集群

voice_type = "BV001_streaming"  # 参考https://www.volcengine.com/docs/6561/97465
host = "openspeech.bytedance.com"
api_url = f"https://{host}/tts_middle_layer/tts"

header = {"Authorization": f"Bearer;{token}"}


def get_online_tts_service(tts_txt='字节跳动语音合成', save_local=True, auto_play=False):
    request_json = {
        "app": {
            "appid": appid,
            "token": "access_token",
            "cluster": cluster
        },
        "user": {
            "uid": "388808087185088"
        },
        "audio": {
            "voice": "other",
            "voice_type": voice_type,
            "encoding": "mp3",
            "speed": 10,
            "volume": 10,
            "pitch": 10
        },
        "request": {
            "reqid": str(uuid.uuid4()),
            "text": tts_txt,
            "text_type": "plain",
            "operation": "query"
        }
    }
    try:
        resp = requests.post(api_url, json.dumps(request_json), headers=header)
        if "data" in resp.json():
            data = resp.json()["data"]
            binary_data = base64.b64decode(data)
            if save_local:
                file_to_save = open("test_submit.mp3", "wb")
                file_to_save.write(binary_data)
            if auto_play:
                # 开始播放
                song = AudioSegment.from_file(io.BytesIO(binary_data), format="mp3")
                play(song)
                pass
            return binary_data
        return None
    except Exception as e:
        logging.exception(e)
        logging.error(f'{tts_txt}')


if __name__ == '__main__':
    count = 1
    s = time.time()
    txt = '你好！我是小艾，我是字节跳动在线语音合成TTS服务'
    for _ in range(count):
        get_online_tts_service(tts_txt=txt,
                               save_local=False, auto_play=True)
    e = time.time()
    print(f'cost time: {(e-s) / count}')