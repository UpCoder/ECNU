#coding=utf-8
"""
识别本地文件
1. 将本地文件上传到字节云 对象存储
2. 获取外链
3. 获取ASR结果
"""
import requests
import json
import time
import os
import os.path

import tos

# Accesskey和Secretkey可在用户火山引擎账号中查找。
ak = "AKLTNGYyNDQxNGY5M2U3NGFkNTliZTc1NmQzOWY1OTUwYWU"
sk = "Wm1NMU5XTXdOamhpTUdZeU5EaGhaR0ZsTVRNMlpqQmtNVE0yTm1FelkyTQ=="
# your endpoint 和 your region 填写Bucket 所在区域对应的Endpoint。# 以华北2(北京)为例，your endpoint 填写 tos-cn-beijing.volces.com，your region 填写 cn-beijing。
endpoint = "tos-cn-beijing.volces.com"
region = "cn-beijing"
bucket_name = "bigfive"
client = tos.TosClientV2(ak, sk, endpoint, region)


def put_file2tos(file_path):
    try:
        # 通过字符串方式添加 Object
        # client.put_object(bucket_name, object_key, content='Hello World')
        resp = client.put_object_from_file(bucket_name, os.path.basename(file_path), file_path,
                                           acl=tos.ACLType.ACL_Public_Read)
        if resp.status_code == 200:
            return os.path.basename(file_path)
        return None
    except tos.exceptions.TosClientError as e:
        # 操作失败，捕获客户端异常，一般情况为非法请求参数或网络异常
        print('fail with client error, message:{}, cause: {}'.format(e.message, e.cause))
    except tos.exceptions.TosServerError as e:
        # 操作失败，捕获服务端异常，可从返回信息中获取详细错误信息
        print('fail with server error, code: {}'.format(e.code))
        # request id 可定位具体问题，强烈建议日志中保存
        print('error with request id: {}'.format(e.request_id))
        print('error with message: {}'.format(e.message))
        print('error with http code: {}'.format(e.status_code))
    except Exception as e:
        print('fail with unknown error: {}'.format(e))
    return None



s = requests

appid = '6747655566'
token = 'M_3Swzuc6aTtP90HE6VHQ58NmBdF_6Rl'
cluster = 'volc_auc_common'
service_url = 'https://openspeech.bytedance.com/api/v1/auc'

headers = {'Authorization': 'Bearer; {}'.format(token)}


def submit_task(audio_url):
    request = {
        "app": {
            "appid": appid,
            "token": token,
            "cluster": cluster
        },
        "user": {
            "uid": "388808087185088_demo"
        },
        "audio": {
            "format": "wav",
            "url": audio_url
        },
        "additions": {
            'with_speaker_info': 'True',
        }
    }

    r = s.post(service_url + '/submit', data=json.dumps(request), headers=headers)
    resp_dic = json.loads(r.text)
    print(resp_dic)
    id = resp_dic['resp']['id']
    print(id)
    return id


def query_task(task_id):
    query_dic = {}
    query_dic['appid'] = appid
    query_dic['token'] = token
    query_dic['id'] = task_id
    query_dic['cluster'] = cluster
    query_req = json.dumps(query_dic)
    print(query_req)
    r = s.post(service_url + '/query', data=query_req, headers=headers)
    print(r.text)
    resp_dic = json.loads(r.text)
    return resp_dic


def file_recognize(audio_url):
    task_id = submit_task(audio_url)
    start_time = time.time()
    lines = []
    while True:
        time.sleep(2)
        # query result
        resp_dic = query_task(task_id)
        if resp_dic['resp']['code'] == 1000: # task finished
            print("success")
            for sentence in resp_dic['resp']['utterances']:
                lines.append(json.dumps(
                    {
                        'spk_id': sentence['additions']['speaker'],
                        'asr_txt': sentence['text'],
                        'start_sec': sentence['start_time'],
                        'end_sec': sentence['end_time']
                    }, ensure_ascii=False
                ) + '\n')
            break
        elif resp_dic['resp']['code'] < 2000: # task failed
            print("failed")
            break
        now_time = time.time()
        if now_time - start_time > 300: # wait time exceeds 300s
            print('wait time exceeds 300s')
            break
    return lines


def handler_pipeline(local_path, save_dir):
    tos_filename = put_file2tos(local_path)
    if tos_filename is None:
        print('file upload to tos failed, please check it!')
        return None
    audio_url = 'https://bigfive.tos-cn-beijing.volces.com/{}'.format(tos_filename)
    lines = file_recognize(audio_url)
    txt_path = os.path.join(save_dir, os.path.basename(os.path.basename(local_path)).split('.')[0] + '.txt')
    with open(txt_path, 'w') as f:
        f.writelines(lines)
    return txt_path


if __name__ == '__main__':
    audio_url = 'https://bigfive.tos-cn-beijing.volces.com/lcy.wav'
    lines = file_recognize(audio_url)
    with open('file_asr.txt', 'w') as f:
        f.writelines(lines)
