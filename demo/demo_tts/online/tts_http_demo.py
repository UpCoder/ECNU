#coding=utf-8

'''
requires Python 3.6 or later
pip install requests
'''
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
        duration = 0    # ms
        ori_data = None
        if "data" in resp.json():
            data = resp.json()["data"]
            ori_data = data
            duration = int(resp.json()['addition']['duration'])
            binary_data = base64.b64decode(data)
            if save_local:
                file_to_save = open("test_submit.mp3", "wb")
                file_to_save.write(binary_data)
            if auto_play:
                # 开始播放
                song = AudioSegment.from_file(io.BytesIO(binary_data), format="mp3")
                play(song)
                pass
            return binary_data, duration, ori_data
        return None, duration, ori_data
    except Exception as e:
        logging.exception(e)
        logging.error(f'{tts_txt}')
        return None, None, None


if __name__ == '__main__':
    count = 1
    s = time.time()
    txt = '你好！我是小艾，很高兴认识你。可以请你做一个自我介绍吗？包括你对自己的描述，兴趣爱好，喜欢的事物等等，你可以先思考一下再开始。每次回答完成后请安静等待5秒，然后会进入下一个问题。'
    for _ in range(count):
        data, duration, ori_data = get_online_tts_service(tts_txt=txt,
                               save_local=False, auto_play=False)
        print(duration)
        print(str(data))
        print(ori_data, type(ori_data))
    e = time.time()
    print(f'cost time: {(e-s) / count}')