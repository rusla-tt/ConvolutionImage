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
        opt = RMSprop(lr=1e-2)
        model.compile(loss='categorical_crossentropy', optimizer=opt)
        return model

    def create_model_rnn(self, iter_num):
        texts = self.m.texts
        for k, num, text in texts:
            for t in text:
                s_text = " ".join(t)
            # t = self.m.get_category(k)
            vocab = self.v.wakachi_vocab(s_text)
            char_indices = dict((c, i) for i, c in enumerate(vocab))
            indices_char = dict((i, c) for i, c in enumerate(vocab))
            maxlen = 10
            step = 3
            sentences = []
            next_chars = []
            for i in range(0, len(text) - maxlen, step):
                sentences.append(vocab[i: i + maxlen])
                next_chars.append(vocab[i + maxlen])
            X = np.zeros((len(sentences), maxlen, len(vocab)), dtype=np.bool)
            y = np.zeros((len(sentences), len(vocab)), dtype=np.bool)
            for i, sentence in enumerate(sentences):
                    for t, char in enumerate(sentence):
                        X[i, t, char_indices[char]] = 1
                        y[i, char_indices[next_chars[i]]] = 1
            model = self.build_rnn(maxlen, vocab)
            collect_dir = 'data/rnn/{}/'.format(k)
            if not os.path.exists(collect_dir):
                os.mkdir(collect_dir)
            for iter in range(1, iter_num):
                model.fit(X, y, batch_size=128, epochs=1, verbose=0)
                model.save(collect_dir + 'model-{}.h5'.format(iter))

    def sample(self, preds, temperature=1.0):
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds+0.0001) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    def prediction_rnn(self, keyword, iter_num):
        t = self.m.get_category(keyword)
        vocab = self.v.wakachi_vocab(t)
        char_indices = dict((c, i) for i, c in enumerate(vocab))
        indices_char = dict((c, i) for i, c in enumerate(vocab))
        maxlen = 10
        step = 3
        sentences = []
        next_chars = []
        for i in range(0, len(t) - maxlen, step):
            sentences.append(vocab[i: i + maxlen])
            next_chars.append(vocab[i + maxlen])
        X = np.zeros((len(sentences), maxlen, len(vocab)), dtype=np.bool)
        y = np.zeros((len(sentences), len(vocab)), dtype=np.bool)
        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                X[i, t, char_indices[char]] = 1
                y[i, char_indices[next_chars[i]]] = 1
        card = random.choice(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
             15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
             32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
             49, 50])
        model = load_model('data/rnn/' + keyword + '/model-{}.h5'.fotmat(card))
        for iteration in range(1, iter_num):
            batch = np.random.randint(1000000, size=1)
            model.fit(X[batch:batch+512],
                      y[batch:batch+512], batch_size=128, nb_epoch=1)
            model.reset_states()
            generated = ''
            start_index = random.randint(0, len(t) - maxlen - 1)
            sentence = t[start_index: start_index + maxlen]
            start = sentence
            generated = ''
            temp = np.random.uniform(0.0, 0.6, size=1)
            try:
                for i in range(200):
                    x = np.zeros((1, maxlen, len(vocab)))
                    for t, char in enumerate(sentence):
                        x[0, t, char_indices[char]] = 1.
                        preds = model.predict(x, verbose=0)[0]
                        next_index = self.sample(preds, temp)
                        next_char = indices_char[next_index]
                        generated += next_char
                        sentence = sentence[1:] + next_char
                        text = '{}{}'.format(start.encode('utf-8'),
                                             generated.encode('utf-8'))
                        text = text.decode('utf-8')
                        texts = text.split(u'\n')
                        if len(texts) > 2:
                            return texts[1]
                        if len(texts) > 3:
                            return texts[2]
                        else:
                            return texts
            except Exception, e:
                continue
