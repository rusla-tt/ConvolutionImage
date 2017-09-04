# -*- encoding: utf-8 -*-
import MeCab
import os
import re
import ngram


class Vacab:

    DIR_BASE_NAME = './data/base_text/'
    TAGGER = ""
    vocablary = {}

    def load_file(self):
        """
        load_file
        return Array<Tuple(Str,Str)>
        """
        texts = []
        dir_list = os.listdir(self.DIR_BASE_NAME)
        for l in dir_list:
            tuple_path = None
            paths = []
            f_list = os.listdir(self.DIR_BASE_NAME+l)
            for f_name in f_list:
                path = None
                path = open(self.DIR_BASE_NAME+l+f_name, 'r')
                path = path.read()
                paths.append(path)
            tuple_path = (l, paths)
            texts.append(tuple_path)
        return texts

    def ngram_vocab(self, gram_num=3, text):
        gram = ngram.Ngram(N=gram_num)
        return gram.ngrams(gram.pad(text))

    def wakachi_vocab(self, text):
        tagger = MeCab.Tagger(self.TAGGER)
        dataset = np.ndarray(len(text), dtype=np.int32)
        for i, word in text:

