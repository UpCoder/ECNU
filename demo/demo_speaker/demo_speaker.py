import logging
import os
import sys
import wave
import json
import numpy as np

from pydub import AudioSegment
from demo.demo_asr.zijie.release_interface import get_client
from vosk import Model, KaldiRecognizer, SpkModel

SPK_MODEL_PATH = "model-spk"
print(os.path.abspath(SPK_MODEL_PATH))
if not os.path.exists(SPK_MODEL_PATH):
    print("Please download the speaker model from "
        "https://alphacephei.com/vosk/models and unpack as {SPK_MODEL_PATH} "
        "in the current folder.")
    sys.exit(1)




def cosine_dist(x, y):
    nx = np.array(x)
    ny = np.array(y)
    return 1 - np.dot(nx, ny) / np.linalg.norm(nx) / np.linalg.norm(ny)


def __handler_sentence(emb, spk_embds, max_speakers=2, threshold=0.5):
    if len(spk_embds) == 0:
        spk_embds.append(emb)
        return 0
    sims = []
    for spk_embd in spk_embds:
        sim = 1 - cosine_dist(emb, spk_embd)
        sims.append(sim)
    max_sims = np.max(sims)
    max_idx = np.argmax(sims)
    if max_sims >= threshold:
        return max_idx
    if len(spk_embds) < max_speakers:
        spk_embds.append(emb)
        return len(spk_embds) - 1
    return max_idx


def handler_wav2sentence(file_name, txt_dir=None):
    spks = []
    count_speaker = 0
    wf = wave.open(file_name, "rb")
    print(wf.getframerate())
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
        print("Audio file must be WAV format mono PCM.")
        sys.exit(1)

    # Large vocabulary free form recognition
    model = Model(lang="cn")
    # model = Model(model_path='C:\\Users\\30644\Downloads\\vosk-model-cn-0.22')
    spk_model = SpkModel(SPK_MODEL_PATH)
    # rec = KaldiRecognizer(model, wf.getframerate(), spk_model)
    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetSpkModel(spk_model)

    start_frame_idx = 0
    end_frame_idx = 0
    records = []
    while True:
        frame_chunk = wf.getframerate() // 4
        data = wf.readframes(frame_chunk)
        if len(data) == 0:
            break
        end_frame_idx += frame_chunk
        if rec.AcceptWaveform(data):
            res = json.loads(rec.Result())
            if "spk" in res:
                # print("X-vector:", res["spk"])
                spk_id = __handler_sentence(res['spk'], spks, max_speakers=2, threshold=0.5)
                line = f'{spk_id}: {start_frame_idx}-{end_frame_idx}' + ' ' + res['text']
                print(line)
                print(f'{spk_id}: {start_frame_idx}({start_frame_idx / wf.getframerate() / 60})'
                      f'-{end_frame_idx}({end_frame_idx / wf.getframerate() / 60})' + ' ' + res['text'])
                records.append(line + '\n')
                start_frame_idx = end_frame_idx
    print("Note that second distance is not very reliable because utterance is too short. "
          "Utterances longer than 4 seconds give better xvector")

    res = json.loads(rec.FinalResult())
    print("Text:", res["text"])
    if "spk" in res:
        spk_id = __handler_sentence(res['spk'], spks, max_speakers=2, threshold=0.5)
        line = f'{spk_id}: {start_frame_idx}-{end_frame_idx}' + ' ' + res['text']
        print(line)
        records.append(line + '\n')

    if txt_dir is None:
        txt_path = os.path.basename(file_name).split('.')[0] + '.txt'
    else:
        txt_path = os.path.join(
            txt_dir,
            os.path.basename(file_name).split('.')[0] + '.txt'
        )
    with open(txt_path, 'w') as f:
        f.writelines(records)
    return txt_path


def merge_asr(cur_asr_result: str, last_asr_result: str):
    """
    合并两句话
    :param cur_asr_result:
    :param last_asr_result:
    :return:
    """
    location_length = 5    # 置信的长度
    end_position = len(last_asr_result)
    while (end_position - location_length) > 0:
        find_str = last_asr_result[end_position-location_length: end_position]
        end_position -= 1
        idx = cur_asr_result.find(find_str)
        if idx == -1:
            continue
        return last_asr_result[:end_position] + cur_asr_result[idx+len(find_str)-1:]
    return cur_asr_result


def calc_metrics_stop_interval(record, stop_duration_threshold_ms, rate, stop_value_threshold):
    """
    计算停顿的次数，和每次停顿的时长
    :param record
    :param stop_duration_threshold_ms:
    :return:
    """
    print('calc_metrics_stop_interval', record, np.max(record), np.min(record))

    cur_record = np.asarray(np.abs(record), np.int16)
    batch_size = rate // 1000  # 代表每ms的数据量
    is_stop = []
    for i in range(int(np.shape(cur_record)[0] // batch_size)):
        cur_slice = cur_record[i * batch_size: (i+1) * batch_size]
        if np.max(cur_slice) <= stop_value_threshold:
            is_stop.append(1)
        else:
            is_stop.append(0)
    print('is_stop length: ', len(is_stop))
    cur_count_stop = 0
    record_stop_durations_ms = []
    for cur_is_stop in is_stop:
        if cur_is_stop == 0:
            record_stop_durations_ms.append(cur_count_stop)
            cur_count_stop = 0
        elif cur_is_stop == 1:
            cur_count_stop += 1
    if cur_count_stop != 0:
        record_stop_durations_ms.append(cur_count_stop)
    record_stop_durations_ms = list(filter(lambda x: x > stop_duration_threshold_ms,
                                           record_stop_durations_ms))
    print('record_stop_durations_ms: ', record_stop_durations_ms)
    return len(record_stop_durations_ms), record_stop_durations_ms


def calc_metrics(duration, binary_data, asr_result, rate):
    data = [{'record': np.frombuffer(binary_data, np.int16)}]

    # data = [self.stop_records[-1]]
    loudness = []
    # 计算高音pitch
    speech_pitch = -np.inf
    speech_pause_count = 0
    speech_pause_duration_ms = []
    for single_record in data:
        # print(single_record[np.where(single_record['record'] >= 0)])
        loudness.append(np.mean(single_record['record']))
        speech_pitch = max(speech_pitch, np.max(single_record['record']))
        # 计算停顿次数
        speech_pause_count_, speech_pause_duration_ms_ = calc_metrics_stop_interval(
            single_record['record'], 300, rate, 3000)
        speech_pause_count += speech_pause_count_
        speech_pause_duration_ms.extend(speech_pause_duration_ms_)
    # 计算语速
    speech_speed = len(asr_result) / (duration * 60)

    # 计算说话时长
    speech_length = duration

    # 计算音调变化
    speech_tone = ''
    metric_dict = {
        'speech_tone': str(speech_tone),  # 音调变化
        # 'speech_pause': str(speech_pause_count),  # 停顿次数
        'speech_loudness': '{:.4f}'.format(np.mean(loudness)),  # 平均响度
        'speech_speed': '{:.4f}'.format(speech_speed),  # 语速
        'speech_pitch': '{:.4f}'.format(speech_pitch),  # 高音pitch
        'speech_length': '{:.4f}'.format(speech_length),  # 回答问题的时长
        # 'speech_pause_duration': '{:.5f}'.format(np.sum(speech_pause_duration_ms))   # 停顿的总时长
    }
    return metric_dict


def handler_sentence2asr(txt_path, wav_path):
    """
    it dependency output of handler_wav2sentence
    """

    def __handler_asr_core(asr_client, cut_data, sound, wf, start_sec, end_sec, max_window_size=30, step_size=3):
        if end_sec - start_sec < max_window_size:
            result = asyncio.run(asr_client.execute_raw(cut_data, wf.getnchannels(), 16, wf.getframerate()))
            if result['payload_msg']['message'] == 'Success':
                cur_asr_result = result['payload_msg']['result'][0]['text']
            else:
                print(f'failed {result}')
                cur_asr_result = ''
        else:
            cur_start_sec = start_sec
            merged_asr_result = None
            while True:
                cur_data = sound[cur_start_sec * 1000: (cur_start_sec + 30) * 1000]
                result = asyncio.run(
                    asr_client.execute_raw(cur_data.raw_data, wf.getnchannels(), 16, wf.getframerate()))
                if result['payload_msg']['message'] == 'Success':
                    cur_asr_result = result['payload_msg']['result'][0]['text']
                else:
                    print(f'{single_data} failed {result}')
                    cur_asr_result = ''
                if merged_asr_result is None:
                    merged_asr_result = cur_asr_result
                else:
                    merged_asr_result = merge_asr(cur_asr_result, merged_asr_result)
                if (cur_start_sec + 30) >= end_sec:
                    break
                else:
                    cur_start_sec += step_size
            cur_asr_result = merged_asr_result
        return cur_asr_result

    import asyncio
    with open(txt_path, 'r') as f:
        records = f.readlines()
    wf = wave.open(wav_path, "rb")
    appid = "6747655566"  # 项目的 appid
    token = "M_3Swzuc6aTtP90HE6VHQ58NmBdF_6Rl"  # 项目的 token
    cluster = "volcengine_streaming_common"  # 请求的集群
    audio_format = "raw"  # wav 或者 mp3，根据实际音频格式设置
    asr_client = get_client(
        {
            'id': 1
        },
        cluster=cluster,
        appid=appid,
        token=token,
        format=audio_format,
        show_utterances=True,
        channel=wf.getnchannels(),
        sample_rate=wf.getframerate()
    )
    # frames = wf.readframes(160000)
    sound = AudioSegment.from_wav(wav_path)
    new_asr_records = []
    last_speaker = None
    start_idx = 0
    last_end_idx = 0
    last_speaker_idx = 0
    last_data = None
    for record in records:
        try:
            print(record)
            single_data = record.split(' ')
            frame_idxs = single_data[1]
            speaker_idxs = single_data[0]
            last_speaker_idx = speaker_idxs
            cur_start_idx, cur_end_idx = frame_idxs.split('-')
            last_end_idx = cur_end_idx
            if last_speaker is None:
                last_speaker = speaker_idxs
            if last_speaker == speaker_idxs:
                last_data = single_data
                continue
            last_speaker = speaker_idxs
            start_sec = int(start_idx) / wf.getframerate()
            end_sec = int(cur_start_idx) / wf.getframerate()
            cut_wav = sound[start_sec * 1000: end_sec * 1000]  # 以毫秒为单位截取[begin, end]区间的音频
            cut_data = cut_wav.raw_data
            cur_asr_result = __handler_asr_core(
                asr_client, cut_data, sound, wf, start_sec, end_sec
            )

            cur_metrics = calc_metrics(end_sec - start_sec, cut_data, cur_asr_result, wf.getframerate())
            new_asr_records.append(json.dumps({
                'spk_id': int(last_data[0][0]),
                'start_sec': start_sec,
                'end_sec': end_sec,
                'asr_txt': cur_asr_result,
                **cur_metrics
            }, ensure_ascii=False) + '\n')
            print(new_asr_records[-1])

            start_idx = cur_start_idx
            last_data = single_data
        except Exception as e:
            logging.exception(e)
            continue
    try:
        start_sec = int(start_idx) / wf.getframerate()
        end_sec = int(last_end_idx) / wf.getframerate()
        cut_wav = sound[start_sec * 1000: end_sec * 1000]  # 以毫秒为单位截取[begin, end]区间的音频
        cut_data = cut_wav.raw_data
        cur_asr_result = __handler_asr_core(
            asr_client, cut_data, sound, wf, start_sec, end_sec
        )
        cur_metrics = calc_metrics(end_sec - start_sec, cut_data, cur_asr_result, wf.getframerate())
        new_asr_records.append(json.dumps({
            'spk_id': last_speaker_idx,
            'start_sec': start_sec,
            'end_sec': end_sec,
            'asr_txt': cur_asr_result,
            **cur_metrics
        }, ensure_ascii=False) + '\n')
        print(new_asr_records[-1])
        with open(txt_path.split('.')[0] + '_asr.txt', 'w') as f:
            f.writelines(new_asr_records)
    except Exception as e:
        logging.exception(e)



if __name__ == '__main__':
    file_name = './lcy.wav'
    txt_path = handler_wav2sentence(file_name)
    handler_sentence2asr(txt_path, file_name)