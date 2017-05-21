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
else:
    from tensorflow_manage_nets.tools import utils
    from tensorflow_manage_nets.nets import net_aencoder as AE


class cnn_ae_lsh:

    def __init__(self, session, npy_convol_path=None, npy_weight_encoders_paths=None, trainable=False, num_class=0, threshold=0):

        if npy_convol_path is not None:
            self.data_dict = np.load(npy_convol_path, encoding='latin1').item()
            print("npy file loaded")
        else:
            self.data_dict = None
            print("random weight")

        self.var_dict = {}
        self.trainable = trainable

        self.weight_paths = npy_weight_encoders_paths
        self.num_class = num_class
        self.AEclass = []
        self.threshold = threshold
        self.sess = session

    def build(self, dim_input, layers=None):

        self.x_batch = tf.placeholder(tf.float32, [None, dim_input])
        # RED VGG ONLY MLP
        self.fc7 = self.fc_layer(self.x_batch, 4096, 4096, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)

        self.fc8 = self.fc_layer(self.relu7, 4096, 10, "fc8")
        self.prob = tf.nn.softmax(self.fc8, name="prob")

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

    # Layer FullConnected
    def fc_layer(self, bottom, in_size, out_size, name, load_weight_force=False):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name, load_weight_force)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
            return fc

    # Generate Parameter FullConnect layer
    def get_fc_var(self, in_size, out_size, name):
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
        weights = self.get_var_fc(initial_value, name, 0, name + "_weights")

        initial_value = tf.truncated_normal([out_size], .0, .001)
        biases = self.get_var_fc(initial_value, name, 1, name + "_biases")

        return weights, biases

    # Construct dictionary with random parameters or load parameters
    def get_var_fc(self, initial_value, name, idx, var_name):

        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
        else:
            value = initial_value

        if self.trainable:
            var = tf.Variable(value, name=var_name)
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)

        self.var_dict[(name, idx)] = var
        assert var.get_shape() == initial_value.get_shape()
        return var
