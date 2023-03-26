import logging
import time
import json
import queue
import threading
import os
from scipy.io import wavfile


def write_wav(data, save_path, fs):
    wavfile.write(save_path, fs, data)


class Recorder(object):

    def __init__(self, save_path='record.txt', wav_dir='./'):
        self.save_path = save_path
        self.q1 = queue.Queue()
        self.wav_dir = wav_dir
        if not os.path.exists(self.wav_dir):
            os.makedirs(self.wav_dir)
            print(f'mkdirs: {self.wav_dir}')
        self.count_face_frame = 0
        self.first_face_timestamp = None

    def record(self):
        with open(self.save_path, 'w') as f:
            while True:
                while not self.q1.empty():
                    try:
                        text_log = self.q1.get()
                        if text_log is None:
                            return
                        print('record:', type(text_log))
                        if type(text_log) == str:
                            f.write(text_log+'\n')
                        elif type(text_log) == dict:
                            if 'timestamp' not in text_log:
                                text_log['timestamp'] = time.time()
                            origin = text_log.get('origin', None)
                            if origin is None or origin == 'body' or origin == 'face':
                                for key in text_log.keys():
                                    text_log[key] = str(text_log[key])
                                text_log_str = json.dumps(text_log)
                                f.write(text_log_str+'\n')
                            elif origin == 'audio':
                                if text_log.get('action', None) == 'save_wav':
                                    filename = '{}.wav'.format(text_log['timestamp'])
                                    save_path = os.path.join(self.wav_dir, filename)
                                    write_wav(
                                        text_log['data'],
                                        save_path,
                                        text_log['fs']
                                    )
                                    text_log['data'] = filename
                                    text_log = json.dumps(text_log)
                                    f.write(text_log + '\n')
                                else:
                                    text_log = json.dumps(text_log)
                                    f.write(text_log + '\n')
                    except Exception as e:
                        logging.exception(e)
                        print(f'text_log: {text_log}')
                time.sleep(0.001)

    def start(self):
        self.record_thread = threading.Thread(target=self.record, args=())
        self.record_thread.start()

    def end(self):
        self.q1.put(None)