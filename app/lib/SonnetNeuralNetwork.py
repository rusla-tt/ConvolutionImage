# -*- encoding: utf-8 -*-
import sonnet as snt
import tensorflow as tf
import sys
import ImagePkl as IP
import configure

"""
Sonnetラッパークラス
提供するメソッドは2種類（create_model, prediction）
"""
class DeepLearning(object):
    
    def __init__(self):
        """
        初期化メソッド
        Yamlファイルをロードする
        加工済みデータのロード
        """
        conf = configure.Configure()
        self.config = conf.load_config()
        pkl = IP.ImagePkl()
        self.images, self.target = pkl.load_dataset()
    
    def create_model(self):
        """
        モデルを作成する為のメソッド
        """
        x = self.images
        y = self.target
        self._class_num = len(set(y))
        perm = np.random.permutation(len(y))
        x = x[perm]
        y = y[perm]
        model = snt.Module(self.build)
        x_traning_predict = model(x, is_training=True,
                keep_prob=tf.constant(0.5))
        x_test_predict = model(x, is_training=False,
                keep_prob=tf.constant(1.0))
        loss = tf.nn.l2_loss(y - x_training_predict)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step = tf.train.GradientDescentOptimizer(
                    learning_rate=1e-2
                    ).minimize(loss)
        with tf.Session() as sess:
            for _ in range(self.config['epoch']):
                sess.run(train_step)
            train_output, train_output_2 = sess.run(
                    [x_training_predict, x_training_predict])
            assert(train_output == train_output_2).all()
            train_output, train_output_2 = sess.run(
                    [x_training_predict, x_test_predict]
                    )
            assert(train_output != train_output_2).any()


    def build(inputs, is_training, keep_prob):
        """
        ネットワークを構築する
        """
        #第1レイヤー(入力)
        outputs = snt.Conv2D(output_channels=96, kernel_shape=9,
                stride=2)(inputs)
        outputs = snt.BatchNorm()(outputs, is_training=is_training)
        outputs = tf.nn.relu(outputs)
        #第2レイヤー(中間層1)
        outputs = snt.Conv2D(output_channels=128,
                kernel_shape=9)(outputs)
        outputs = snt.BatchNorm()(outputs, is_training=is_training)
        outputs = tf.nn.relu(outputs)
        outputs = self._max_pooling_2x2(outputs)
        #第3レイヤー(中間層2)
        outputs = snt.Conv2D(output_channels=256, kernel_shape=9,
                stride=2)(outputs)
        outputs = snt.BatchNorm()(outputs, is_training=is_training)
        outputs = tf.nn.relu(outputs)
        outputs = self._max_pooling_2x2(outputs)
        outputs = snt.BatchFlatten()(outputs)
        #第4レイヤー(中間層3)
        outputs = snt.Linear(output_size=1024)(outputs)
        outputs = tf.nn.relu(outputs)
        outputs = tf.nn.dropout(outputs, keep_prob=keep_prob)
        #第5レイヤー(出力)
        outputs = snt.Linear(output_size=self._class_num)(output)
        return outputs

    def _max_pooling_2x2(self, output):
        return tf.nn.max_pool(output, ksize=[1, 2, 2, 1],
                stride=[1, 2, 2, 1], padding='SAME')

    def prediction(self, img):
        """
        学習モデルをロードしてpredictionする為のモデル
        """
        pass
