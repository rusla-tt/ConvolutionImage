# -*- encoding: utf-8 -*-
import MeCab
import os
import ngram
import configure
import csv


class Vocab:

    def __init__(self):
        config = configure.Configure()
        conf = config.load_config()
        self.DIR_BASE_NAME = conf['vocab_dir_base_name']
        self.TAGGER = conf['vocab_tagger']
        self.vocablary = {}
        self.f = open('./data/map/map.csv', 'wb')
        self.wt = csv.writer(self.f)

    def load_file(self):
        """
        load_file
        return Array<Tuple(Str,Str)>
        """
        texts = []
        dir_list = os.listdir(self.DIR_BASE_NAME)
        count = 0
        for l in dir_list:
            tuple_path = None
            paths = []
            f_list = os.listdir(self.DIR_BASE_NAME+l)
            for f_name in f_list:
                path = None
                path = open(self.DIR_BASE_NAME+l+"/"+f_name, 'r')
                path = path.read()
                paths.append(path)
            tuple_path = (str(count), paths)
            texts.append(tuple_path)
            self.wt.writerow([str(count), l])
            count = count + 1
        self.f.close()
        return texts

    def ngram_vocab(self, text, gram_num=3):
        gram = ngram.Ngram(N=gram_num)
        gram = gram.ngrams(gram.pad(text))
        return gram

    def wakachi_vocab(self, text):
        tagger = MeCab.Tagger(self.TAGGER)
        wordlist = tagger.parseToNode(text)
        words = self._get_surfaces(wordlist)
        return words

    def _get_surfaces(self, node):
        words = []
        while node:
            word = node.surface
            words.append(word)
            node = node.next
        return words
