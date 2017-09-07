# -*- encoding: utf-8 -*-
import matplotlib.pyplot as plt
import os
plt.style.use("ggplot")
import numpy as np
import cv2 as cv
import configure
import json
import csv
import random
from keras.callbacks import EarlyStopping
from keras.callbacks import LearningRateScheduler
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.models import Sequential, model_from_json
from keras.optimizers import SGD
from ImagePkl import ImagePkl
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.models import load_model
import Marcov
import Vocab
import unicodedata


class Schedule(object):
    def __init__(self, init=0.01):
        self.init = init

    def __call__(self, epoch):
        lr = self.init
        for i in xrange(1, epoch+1):
            if i % 5 == 0:
                lr *= 0.5
        return lr


class DeepLearning(object):
    def __init__(self):
        config = configure.Configure()
        conf = config.load_config()
        self._model = None
        self.m = Marcov.Marcov()
        self.v = Vocab.Vocab()
        self.DIR_BASE_NAME = conf['vocab_dir_base_name']
        self.img_rows = conf['img_rows']
        self.img_cols = conf['img_cols']
        self.img_channels = conf['img_channels']
        self._model_json_path = conf['model_json_path']
        self._model_weight_path = conf['model_weight_path']
        try:
            self._model = self._load_model()
        except:
            pass

    def get_schedule_func(self, init):
        return Schedule(init)

    def build(self, num_classes=2):
        model = Sequential()
        model.add(Convolution2D(96, (3, 3), padding='same',
                                input_shape=(self.img_channels, self.img_rows,
                                             self.img_cols)))
        model.add(Activation('relu'))
        model.add(Convolution2D(128, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
        model.add(Convolution2D(256, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Convolution2D(256, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(1, 1)))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))
        return model

    def create_model(self):
        dataset = ImagePkl()
        data, target = dataset.load_dataset()
        n_class = len(set(target))
        perm = np.random.permutation(len(target))
        x = data[perm]
        y = target[perm]
        model = self.build(n_class)
        init_learning_rate = 1e-2
        opt = SGD(lr=init_learning_rate, decay=0.0, momentum=0.9,
                  nesterov=False)
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=opt, metrics=["acc"])
        early_stopping = EarlyStopping(monitor='val_loss',
                                       patience=3, verbose=0, mode='auto')
        lrs = LearningRateScheduler(self.get_schedule_func(init_learning_rate))
        model.fit(x, y,
                  batch_size=128,
                  epochs=200, validation_split=0.1,
                  verbose=1,
                  callbacks=[early_stopping, lrs])
        model.save(self._model_weight_path)
        open(self._model_json_path, 'wb').write(model.to_json())
        csv.writer(open('./models/label.csv', 'wb')).writerow(list(set(y)))

    def _load_model(self):
        # dataset = ImagePkl()
        # data, target = dataset.load_dataset()
        # n_class = len(set(target))
        # perm = np.random.permutation(len(traget))
        # x = data[perm]
        # y = target[perm]
        model = model_from_json(
            open(self._model_json_path).read())
        model.load_weights(self._model_weight_path)
        init_learning_rate = 1e-2
        opt = SGD(lr=init_learning_rate, decay=0.0, momentum=0.9,
                  nesterov=False)
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=opt, metrics=["acc"])
        return model

    def prediction(self):
        image = cv.imread('./tmp/tmp.jpg')
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB).astype(np.float32)
        image = cv.resize(image, (self.img_rows, self.img_cols))
        image = image.transpose(2, 0, 1)
        image = image.reshape((1,) + image.shape)
        image = image/255.
        model = self._load_model()
        result = model.predict(image)
        label, ratio = self._ratio_label(result)
        return [label, ratio]

    def _ratio_label(self, ratio_list):
        labels = csv.reader(open('./models/label.csv', 'rb')).next()
        code_list = json.loads(open('./models/code_name.json', 'rb').read())
        tmp_list = []
        for l, r in zip(labels, ratio_list[0]):
            tmp_list.append([l, r])
        ratio = 0.0
        for t in tmp_list:
            label = ""
            if t[1] > ratio:
                ratio = t[1]
                code = t[0]
        return code_list[str(code)], ratio

    def build_rnn(self, maxlen, vocab):
        model = Sequential()
        model.add(LSTM(128, input_shape=(maxlen, len(vocab))))
        model.add(Dense(len(vocab)))
        model.add(Activation('softmax'))
        opt = RMSprop(lr=0.01)
        model.compile(loss='categorical_crossentropy', optimizer=opt)
        return model

    def create_model_rnn(self, iter_num):
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
            tuple_path = (l, str(count), paths)
            texts.append(tuple_path)
            count = count + 1
        for k, num, text in texts:
            s_text = " ".join(text)
            # t = self.m.get_category(k)
            vocab = self.v.wakachi_vocab(s_text)
            words = sorted(list(set(vocab)))
            char_indices = dict((c, i) for i, c in enumerate(words))
            indices_char = dict((i, c) for i, c in enumerate(words))
            maxlen = 10
            step = 3
            sentences = []
            next_chars = []
            for i in range(0, len(vocab) - maxlen, step):
                sentences.append(vocab[i: i + maxlen])
                next_chars.append(vocab[i + maxlen])
            X = np.zeros((len(sentences), maxlen, len(words)), dtype=np.bool)
            y = np.zeros((len(sentences), len(words)), dtype=np.bool)
            for i, sentence in enumerate(sentences):
                for t, char in enumerate(sentence):
                    X[i, t, char_indices[char]] = 1
                y[i, char_indices[next_chars[i]]] = 1
            model = self.build_rnn(maxlen, words)
            collect_dir = 'data/rnn/{}/'.format(k)
            if not os.path.exists(collect_dir):
                os.mkdir(collect_dir)
            if len(s_text) != 0:
                print collect_dir
                try:
                    for iter in range(0, iter_num):
                        model.fit(X, y, batch_size=128, epochs=10, verbose=0)
                        model.save(collect_dir + 'model-{}.h5'.format(str(iter)))
                except:
                    continue

    def sample(self, preds, temperature=1.0):
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds+0.0001) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    def prediction_rnn(self, keyword, iter_num, card_num=50):
        texts = self.m.texts
        t = None
        keyword_conv = unicodedata.normalize(
            'NFC', keyword.decode('utf-8')).encode('utf-8')
        for k, num, text in texts:
            k_conv = unicodedata.normalize(
                'NFC', k.decode('utf-8')).encode('utf-8')
            if k_conv == keyword_conv:
                t = " ".join(text)
        if t is None:
            raise Exception
        vocab = self.v.wakachi_vocab(t)
        words = sorted(list(set(vocab)))
        char_indices = dict((c, i) for i, c in enumerate(words))
        indices_char = dict((i, c) for i, c in enumerate(words))
        maxlen = 10
        step = 3
        sentences = []
        next_chars = []
        for i in range(0, len(vocab) - maxlen, step):
            sentences.append(vocab[i: i + maxlen])
            next_chars.append(vocab[i + maxlen])
        X = np.zeros((len(sentences), maxlen, len(words)), dtype=np.bool)
        y = np.zeros((len(sentences), len(words)), dtype=np.bool)
        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                X[i, t, char_indices[char]] = 1
            y[i, char_indices[next_chars[i]]] = 1
        card_number = []
        for i in range(0, card_num):
            card_number.append(i)
        card = random.choice(card_number)
        model = load_model(
            'data/rnn/' + keyword_conv + '/model-{}.h5'.format(card))
        for iteration in range(0, iter_num):
            batch = np.random.randint(100000, size=1)[0]
            model.fit(X[batch: batch+512],
                      y[batch: batch+512],
                      batch_size=128, epochs=1, verbose=0)
            model.reset_states()
            generated = ''
            start_index = random.randint(0, len(words) - maxlen - 1)
            sentence = vocab[start_index: start_index + maxlen]
            start = sentence
            generated = ''
            temp = np.random.uniform(0.0, 0.6, size=1)
            try:
                texts = "".join(start)
                for i in range(100):
                    x = np.zeros((1, maxlen, len(words)))
                    for t, char in enumerate(sentence):
                        x[0, t, char_indices[char]] = 1.
                    preds = model.predict(x, verbose=0)[0]
                    next_index = self.sample(preds, temp)
                    next_char = indices_char[next_index]
                    generated += next_char
                    sentence = " ".join(sentence[1:])
                    sentence = sentence + " " + next_char
                    sentence = sentence.split(" ")
                    texts = texts + next_char
                return texts
            except Exception, e:
                print e
                continue
