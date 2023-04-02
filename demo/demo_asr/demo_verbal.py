# coding=utf-8

import time
import jieba.posseg as pseg
from cnsenti import Sentiment
from cnsenti import Emotion


class verbal_analyzer(object):
    def __init__(self):
        self.cnsenti_senti = Sentiment()
        self.cnsenti_emotion = Emotion()

        self.stop_sign = ['。', '？', '！']
        self.save_word_attr = ['n', 'v', 'a', 'r', 'd', 'e', 't', 'w']
        # n名词 v动词 a形容词 d副词 r代词 e叹词 t时间词 w标点符号

        self.emo_map = {
            "good_word": "好",
            "happy_word": "乐",
            "sad_word": "哀",
            "angry_word": "怒",
            "fear_word": "惧",
            "disgust_word": "恶",
            "surprise_word": "惊"
        }

    def run(self, text):
        text_attr = {
            "word_num": 0,
            "sentence_num": 0,
            "pos_word_num": 0,
            "neg_word_num": 0
        }
        words = pseg.cut(text)
        for word, word_attr in words:
            text_attr["word_num"] += 1
            if word in self.stop_sign:
                text_attr["sentence_num"] += 1
            if word_attr in self.save_word_attr:
                if word_attr not in text_attr:
                    text_attr[word_attr] = 0
                text_attr[word_attr] += 1

        result = self.cnsenti_senti.sentiment_count(text)
        text_attr['pos_word_num'] = result['pos']
        text_attr['neg_word_num'] = result['neg']

        result = self.cnsenti_emotion.emotion_count(text)
        for k, v in self.emo_map.items():
            text_attr[k] = result[v]
        return text_attr


if __name__ == '__main__':
    text = "哈喽，你好，我是本次访谈的访谈者。我叫小熊啊，很开心能够参与本次访谈，我有点紧张。"

    ta = verbal_analyzer()
    start_time = time.time()
    ta.run("text")
    start_time = time.time()
    rtn = ta.run(text)
    print(time.time() - start_time)
    print(rtn)