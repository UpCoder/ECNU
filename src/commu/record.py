import time
import json
import queue
import threading

class Recorder(object):

    def __init__(self, save_path='record.txt'):
        self.save_path = save_path
        self.q1 = queue.Queue()

    def record(self):
        with open(self.save_path, 'w') as f:
            while True:
                while not self.q1.empty():
                    text_log = self.q1.get()
                    if text_log is None:
                        return
                    print('record:', type(text_log))
                    if type(text_log) == str:
                        f.write(text_log+'\n')
                    elif type(text_log) == dict:
                        text_log['timestamp'] = time.time()
                        text_log = json.dumps(text_log)
                        f.write(text_log+'\n')
                time.sleep(0.01)

    def start(self):
        self.record_thread = threading.Thread(target=self.record, args=())
        self.record_thread.start()

    def end(self):
        self.q1.put(None)