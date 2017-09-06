# -*- encoding: utf-8 -*-

import Vocab
import re
import random
import csv
import unicodedata


class Marcov:
    """
    マルコフ連鎖で文章を生成するためのクラス
    """
    def __init__(self):
        self.v = Vocab.Vocab()
        self.texts = self.v.load_file()

    def get_category(self, key):
        f = open("data/map/map.csv", "rb")
        reader = csv.reader(f)
        number = None
        s_text = None
        key_convert = unicodedata.normalize('NFC',
                                            key.decode('utf-8')).encode('utf-8')
        for r in reader:
            r_1_convert = unicodedata.normalize('NFC',
                                                r[1].decode('utf-8')).encode('utf-8')
            if key_convert == r_1_convert:
                number = r[0]
        for k, text in self.texts:
            if number == k:
                s_text = " ".join(text)
                return s_text
        if s_text is None:
            raise Exception

    def marcov_main(self, keys, ngram_mode=False, word_length=90):
        text = self.get_category(keys)
        vocab = None
        if ngram_mode:
            vocab = self.v.ngram_vocab(text)
        else:
            vocab = self.v.wakachi_vocab(text)
        marcov = {}
        tmp_word1 = ""
        tmp_word2 = ""
        tmp_word3 = ""
        for n in vocab:
            if tmp_word1 and tmp_word2 and tmp_word3:
                if not marcov.has_key(tmp_word3):
                    marcov[tmp_word3] = {}
                if not marcov[tmp_word3].has_key(tmp_word2):
                    marcov[tmp_word3][tmp_word2] = {}
                if not marcov[tmp_word3][tmp_word2].has_key(tmp_word1):
                    marcov[tmp_word3][tmp_word2][tmp_word1] = []
                marcov[tmp_word3][tmp_word2][tmp_word1].append(n)
            tmp_word3 = tmp_word2
            tmp_word2 = tmp_word1
            tmp_word1 = n
        sentence = ""
        count = 0
        choice_words3 = random.choice(marcov.keys())
        choice_words2 = random.choice(marcov[choice_words3].keys())
        choice_words1 = random.choice(
            marcov[choice_words3][choice_words2].keys())
        sentence = choice_words3 + choice_words2 + choice_words1
        while count < word_length:
            choice_tmp = random.choice(
                marcov[choice_words3][choice_words2][choice_words1])
            sentence += choice_tmp
            choice_words3 = choice_words2
            choice_words2 = choice_words1
            choice_words1 = choice_tmp
            count += 1
            sentence = sentence.split(" ", 1)[0]
            ng_key = re.compile("[!-/:-@[-`{-~]")
            sentences = ng_key.sub(" ", sentence)
        words = re.sub(re.compile("[!-~]"), "", sentences)
        return words
