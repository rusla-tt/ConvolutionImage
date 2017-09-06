# -*- encoding: utf-8 -*-
import MeCab
import os
import ngram
import configure


class Vocab:

    def __init__(self):
        config = configure.Configure()
        conf = config.load_config()
        self.DIR_BASE_NAME = conf['vocab_dir_base_name']
        self.TAGGER = conf['vocab_tagger']
        self.vocablary = {}

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

    def ngram_vocab(self, text, gram_num=3):
        gram = ngram.Ngram(N=gram_num)
        gram = gram.ngrams(gram.pad(text))
        return gram

    def wakachi_vocab(self, text):
        tagger = MeCab.Tagger(self.TAGGER)
        wordlist = tagger.parse(text)
        return wordlist
