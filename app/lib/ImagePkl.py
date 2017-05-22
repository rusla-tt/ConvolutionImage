# -*- encoding: utf-8 -*-
import os
import six.moves.cPickle as pickle
import numpy as np
import configure
import json
try:
    import cv2 as cv
except:
    raise

class ImagePkl:
    # 初回
    json_list = {}

    def __init__(self):
        cfg = configure.Configure()
        self._configure = cfg.load_config()
        self._data_dir_path = self._configure['data_dir_path']
        self._n_types_target = -1
        self._dump_name = self._configure['pkl_dump_file_name']
        self._image_size = self._configure['image_size']

    # data set 用画像が保存されているディレクトリを検索する
    def get_dir_list(self):
        tmp = os.listdir(self._data_dir_path)
        if tmp is None:
            return None
        ret = []
        for x in tmp:
            if os.path.isdir(self._data_dir_path+x):
                if len(os.listdir(self._data_dir_path+x)) >= 2:
                    ret.append(x)
        return sorted(ret)

    # class id の取得
    def get_class_id(self, fname):
        dir_list = self.get_dir_list()
        dir_name = filter(lambda x:x in fname, dir_list)
        return dir_list.index(dir_name[0])

    def get_class_name(self, id):
        dir_list = self.get_dir_list()
        return dir_list[id]

    def set_code_name(self, name, id):
        self.json_list[str(id)] = name

    def create_data_target(self):
        dir_list = self.get_dir_list()
        ret = {}
        target = []
        data = []
        print("create pkl data")
        for i, dir_name in enumerate(dir_list):
            file_list = os.listdir(self._data_dir_path + dir_name)
            for file_name in file_list:
                root, ext = os.path.splitext(file_name)
                if ext.upper() == '.JPG':
                    abs_name = self._data_dir_path + dir_name + '/' + file_name
                    class_id = self.get_class_id(abs_name)
                    self.set_code_name(dir_name, class_id)
                    target.append(class_id)
                    image = cv.imread(abs_name)
                    image = cv.resize(image, (self._image_size, self._image_size))
                    image = image.transpose(2, 0, 1)
                    image = image/255.
                    data.append(image)
        data = np.array(data, np.float32)
        target = np.array(target, np.int32)
        self._dump_dataset(data, target)
        self._dump_json_code()
        print("done")

    def _dump_json_code(self):
        json.dump(self.json_list,
                open('./models/code_name.json', 'wb'),
                ensure_ascii=False)

    def _dump_dataset(self, data, target):
        pickle.dump((data, target), open(self._dump_name, 'wb'), -1)

    def load_dataset(self):
        data, target = pickle.load(open(self._dump_name, 'rb'))
        return data, target
