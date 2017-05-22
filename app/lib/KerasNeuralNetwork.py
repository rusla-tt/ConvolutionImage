# -*- encoding: utf-8 -*-
import keras
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import numpy as np
import pandas as pd
import cv2 as cv
import configure
import h5py
import json
import csv
from keras.callbacks import EarlyStopping
from keras.callbacks import LearningRateScheduler
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, model_from_json
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.utils import np_utils
from ImagePkl import ImagePkl
from keras.models import load_model

class Schedule(object):
    def __init__(self, init=0.01):
        self.init = init

    def __call__(self, epoch):
        lr =self.init
        for i in xrange(1, epoch+1):
            if i%5==0:
                lr *= 0.5
        return lr

class DeepLearning(object):
    def __init__(self):
        config = configure.Configure()
        conf = config.load_config()
        self._model = None
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
        #dataset = ImagePkl()
        #data, target = dataset.load_dataset()
        #n_class = len(set(target))
        #perm = np.random.permutation(len(traget))
        #x = data[perm]
        #y = target[perm]
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
