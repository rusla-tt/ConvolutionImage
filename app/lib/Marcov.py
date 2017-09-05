# -*- encoding: utf-8 -*-

import Vocab
import re

class Marcov:
    """
    マルコフ連鎖で文章を生成するためのクラス
    """
    def __init__(self):
        self.v = Vocab.Vocab()
        self.texts = self.v.load_file()

    def _get_category(self, key):
        for k, text in self.texts:
            if k == key:
                s_text = " ".join(text)
                return s_text

    def marcov_main(self, key, ngram_mode=False, word_length=90):
        text = self._get_category(key)
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
            if tmp_word and tmp_word2 and tmp_word3:
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
        choice_words1 = random.choice(marcov[choice_words3][choice_words2].keys())
        sentence = choice_words3 + choice_word2 + choice_word1
        while count < word_length:
            choice_tmp = random.choice(marcov[choice_word3][choice_word2][choice_word1])
            sentence += choice_tmp
            choice_word3 = choice_word2
            choice_word2 = choice_word1
            choice_word1 = choice_tmp
            count += 1
            sentence = sentence.split(" ",1)[0]
            ng_key = re.compile("[!-/:-@[-`{-~]")
            sentences = ng_key.sub(sentence)
        words = re.sub(re.comile("[!-~]"), "", sentences)
        return words
