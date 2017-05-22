# -*- encoding: utf-8 -*-
import yaml


class Configure():

    def load_config(self):
        path = 'config/config.yaml'
        f = open(path, 'rb')
        data = yaml.load(f)
        return data
