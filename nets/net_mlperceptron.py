import tensorflow as tf
import time
import numpy as np


class MLPerceptron:
    """
    A trainable version VGG19.
    """

    def __init__(self, mlp_npy_path=None, trainable=True, learning_rate=0.05, dropout=0.5):
        if mlp_npy_path is not None:
            self.data_dict = np.load(mlp_npy_path, encoding='latin1').item()
            print("npy file loaded")
        else:
            self.data_dict = None
            print("random weight")

        self.var_dict = {}
        self.trainable = trainable
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.net = {}

    def build(self, input_batch, target, train_mode=None, layers=[4096, 2048, 2]):

        start_time = time.time()
        current_input = input_batch

        #
        # BUILD THE ENCODER
        # -----------------
        idx = 0
        for i, n_output in enumerate(layers[:-1]):
            n_input = current_input.get_shape().as_list()[1]
            name = 'fc_' + str(i)
            self.net[name] = tf.nn.relu(self.fc_layer(current_input, n_input, n_output, name))
            # DROPOUT
            if self.trainable is True:
                self.net[name] = tf.cond(train_mode, lambda: tf.nn.dropout(self.net[name], self.dropout), lambda: self.net[name])

            current_input = self.net[name]
            idx = i

        name = 'fc_' + str(idx+1)
        n_input = current_input.get_shape().as_list()[1]
        n_output = layers[-1]
        self.net[name] = self.fc_layer(current_input, n_input, n_output, name)
        self.net['prob'] = tf.nn.softmax(self.net[name], name="prob")

        # COST - TRAINING
        self.cost = tf.reduce_mean((self.net['prob'] - target) ** 2)
        self.train = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)
        # self.train = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)

        self.data_dict = None
        print(("build model finished: %ds" % (time.time() - start_time)))

    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
            return fc

    def get_fc_var(self, in_size, out_size, name):
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
        weights = self.get_var_fc(initial_value, name, 0, name + "_weights")

        initial_value = tf.truncated_normal([out_size], .0, .001)
        biases = self.get_var_fc(initial_value, name, 1, name + "_biases")

        return weights, biases

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

    def save_npy(self, sess, npy_path="./mlp-save.npy"):
        assert isinstance(sess, tf.Session)

        data_dict = {}

        for (name, idx), var in list(self.var_dict.items()):
            var_out = sess.run(var)
            if name not in data_dict:
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)
        print("File saved", npy_path)
        return npy_path
