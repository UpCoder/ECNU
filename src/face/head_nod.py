from queue import Queue

class NodDetecter(object):

    def __init__(self, nod_threshold):
        self.q = Queue()
        self.size = 50
        self.nod_threshold = nod_threshold

    def update(self, val):
        self.q.put(val)
        while self.q.qsize() > self.size:
            self.q.get()

    def clear(self):
        while not self.q.empty():
            self.q.get()

    def detect(self):
        nod_flag = False
        v_list = list(self.q.queue)

        min_v = min(v_list)
        min_index = v_list.index(min_v)

        max_v = max(v_list[:(min_index+1)])
        last_v = v_list[-1]
        if last_v - min_v >= self.nod_threshold and max_v > last_v:
            nod_flag = True
        return nod_flag