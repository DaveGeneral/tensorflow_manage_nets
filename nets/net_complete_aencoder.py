import time
import tensorflow as tf
import sys, os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

switch_server = True

testdir = os.path.dirname('__file__')
srcdir = '..'
sys.path.insert(0, os.path.abspath(os.path.join(testdir, srcdir)))

if switch_server is True:
    from tools import utils
    from nets import net_aencoder as AE
    from tools.dataset_csv import Dataset_csv
else:
    from tensorflow_manage_nets.tools import utils
    from tensorflow_manage_nets.nets import net_aencoder as AE
    from tensorflow_manage_nets.tools.dataset_csv import Dataset_csv


class ae_multiClass:

    def __init__(self, session, npy_weight_paths=None, num_class=0, threshold=0):

        self.weight_paths = npy_weight_paths
        self.num_class = num_class
        self.AEclass = []
        self.threshold = threshold
        self.sess = session

    def build(self, dim_input, layers):

        self.x_batch = tf.placeholder(tf.float32, [None, dim_input])

        for i in range(self.num_class):
            self.AEclass.append(AE.AEncoder(self.weight_paths[i]))
            self.AEclass[i].build(self.x_batch, layers)

        self.sess.run(tf.global_variables_initializer())

    def search_sample(self, sample):

        y_result = []
        for i in range(len(sample)):
            x_ = [sample[i]]
            cost_class = []
            for class_i in range(self.num_class):
                cost_i = self.sess.run(self.AEclass[class_i].cost, feed_dict={self.x_batch: x_})
                cost_class.append(cost_i)

            cost_class = np.array(cost_class)
            res = np.argsort(cost_class)
            y_result.append([res, cost_class[res]])

        return y_result

