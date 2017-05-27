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


class cnn_ae_lsh:

    def __init__(self, session,
                 npy_convol_path=None,
                 npy_ae_path=None,
                 npy_ae_class_paths=None,
                 normal_max_path = None,
                 trainable=False,
                 num_class=0,
                 k_classes=1,
                 threshold=0):

        if npy_convol_path is not None:
            self.data_dict = np.load(npy_convol_path, encoding='latin1').item()
            print("npy file loaded", npy_convol_path)
        else:
            self.data_dict = None
            print("random weight")

        if normal_max_path is not None:
            self.normalization_max = utils.load_max_csvData(normal_max_path)
        else:
            self.normalization_max = 1
            print("Data no normalization")

        self.var_dict = {}
        self.trainable = trainable
        self.weight_ae_path = npy_ae_path
        self.weight_ae_class_paths = npy_ae_class_paths
        self.num_class = num_class
        self.AEclass = []
        self.sess = session

        self.k = k_classes
        self.threshold = threshold

    def build(self, dim_input, layers=None):

        start_time = time.time()
        print("build model started")
        self.x_batch = tf.placeholder(tf.float32, [None, dim_input])

        # ----------------
        # NET VGG ONLY MLP

        self.fc1 = self.fc_layer(self.pool2, 800, 500, "ip1")
        self.relu1 = tf.nn.relu(self.fc1)
        self.fc2 = self.fc_layer(self.relu1, 500, 100, "latent")
        self.relu2 = tf.nn.relu(self.fc2)
        self.fc3 = self.fc_layer(self.relu2, 100, 10, "ip2")
        self.probVGG = tf.nn.softmax(self.fc3, name="prob")

        # ------------------
        # AUTOENCODER GLOBAL

        self.AEGlobal = AE.AEncoder(self.weight_ae_path)
        self.AEGlobal.build(self.x_batch, layers)

        # ---------------------
        # AUTOENCODERS BY CLASS

        for i in range(self.num_class):
            self.AEclass.append(AE.AEncoder(self.weight_ae_class_paths[i]))
            self.AEclass[i].build(self.x_batch, layers)

        self.sess.run(tf.global_variables_initializer())
        print(("build model finished: %ds" % (time.time() - start_time)))

    # TEST VGG
    def test_vgg(self, objData, normalize=False):

        if normalize is True:
            objData.normalization(self.normalization_max)

        count_success = 0
        prob_predicted = []
        plot_predicted = []

        label_total = []
        prob_total = np.random.random((0, self.num_class))

        print('\n# TEST VGG TRAINED')
        for i in range(objData.total_batchs_complete):
            batch, label = objData.generate_batch()
            prob = self.sess.run(self.probVGG, feed_dict={self.x_batch: batch})

            label_total = np.concatenate((label_total, label), axis=0)
            prob_total = np.concatenate((prob_total, prob), axis=0)

            # Acumulamos la presicion de cada iteracion, para despues hacer un promedio
            count, prob_predicted, plot_predicted = utils.process_prob(label, prob, predicted=prob_predicted,
                                                                       plot_predicted=plot_predicted)
            count_success = count_success + count
            objData.next_batch_test()

        # promediamos la presicion total
        print('\n# STATUS:')
        y_true = objData.labels
        y_prob = prob_predicted
        utils.metrics_multiclass(y_true, y_prob)

    # TEST AUTO-ENCODER
    def test_ae_global(self, objData, normalize=False):

        if normalize is True:
            objData.normalization(self.normalization_max)

        total = objData.total_inputs
        cost_total = 0

        print('\n# TEST AENCODER TRAINED')
        for i in range(objData.total_batchs_complete):
            x_, label = objData.generate_batch()
            cost = self.sess.run(self.AEGlobal.cost, feed_dict={self.x_batch: x_})
            cost_total = cost_total + cost
            objData.next_batch_test()

        print(cost_total, cost_total / total)

    # TEST AUTO-ENCODER BY CLASS
    def test_ae_class(self, objData, normalize=False):

        if normalize is True:
            objData.normalization(self.normalization_max)

        minibatch_aux = objData.minibatch
        objData.change_minibatch(1)

        y_true = objData.labels
        y_result = []

        print('\n# TEST AENCODER BY CLASS TRAINED')
        for ix in range(objData.total_batchs_complete):
            x_, label = objData.generate_batch()
            cost_class = []
            for class_i in range(self.num_class):
                cost_i = self.sess.run(self.AEclass[class_i].cost, feed_dict={self.x_batch: x_})
                cost_class.append(cost_i)

            y_result.append(np.argsort(cost_class)[0])
            objData.next_batch_test()

        objData.change_minibatch(minibatch_aux)
        utils.metrics_multiclass(y_true, y_result)


    # GENERATE DATA ENCODE
    def generate_data_encode_matrix(self, data, normalize=True):

        if normalize is True:
            data = data / self.normalization_max

        total = len(data)
        result = []

        print('\n# GENERATE DATA ENCODE')
        for i in range(total):
            x_ = [data[i]]
            layer = self.sess.run(self.AEGlobal.z, feed_dict={self.x_batch: x_})
            result.append(layer[0])
        return result

    # GENERATE DATA ENCODE
    def generate_data_encode(self, objData, normalize=False, csv_save=False, path_save='', csv_name='encode'):
        if normalize is True:
            objData.normalization(self.normalization_max)

        total = objData.total_inputs
        cost_total = 0
        result = []

        if csv_save is True:
            print('\n# GENERATE DATA ENCODE')
            for i in range(objData.total_batchs_complete):
                x_, label = objData.generate_batch()
                cost, layer = self.sess.run([self.AEGlobal.cost, self.AEGlobal.net['encodeFC_1']], feed_dict={self.x_batch: x_})
                utils.save_layer_output(layer, label, name=csv_name, dir=path_save)

                cost_total = cost_total + cost
                objData.next_batch_test()

            print(cost_total, cost_total / total)

        else:
            print('\n# GENERATE DATA ENCODE')
            for i in range(objData.total_batchs_complete):
                x_, label = objData.generate_batch()
                layer = self.sess.run(self.AEGlobal.z, feed_dict={self.x_batch: x_})
                objData.next_batch_test()
                result.append(np.concatenate((layer[0], [label]), axis=0))

            return result


    # TEST SEARCH
    def search_sample(self, sample):

        encode_result = []
        for i in range(len(sample)):
            x_ = [sample[i]]

            probability = self.sess.run(self.probVGG, feed_dict={self.x_batch: x_})
            probability = np.array(probability[0])

            # Clases elejidas y % de representacion
            clss = np.argsort(probability)[::-1][:self.k]
            result = probability[clss]

            # Normalizamos la data para los Auto-encoders
            assert np.shape(x_[0]) == np.shape(self.normalization_max), print('Different', np.shape(x_[0]),
                                                                              np.shape(self.normalization_max))
            x_ = x_ / self.normalization_max

            # Ingresamos la muestra original al AE-GLOBAL, obtenemo un vector de 512 + clase y probabilidad [-1,-1]
            encode_list = []
            layer = self.sess.run(self.AEGlobal.z, feed_dict={self.x_batch: x_})
            encode_list.append(np.concatenate((layer[0], [-1, -1]), axis=0))

            # Obtenemos la codificacion de la muestra original a traves de la AE-By-CLASS, las k codificaciones de 512
            for j in range(len(clss)):
                # Pasamos la muestra original por los AE de las clases elegidas
                layer = self.sess.run(self.AEclass[clss[j]].z, feed_dict={self.x_batch: x_})
                # La muestra reconstruida, se concatena con su numero de clase y su valor de probabilidad
                encode = np.concatenate((layer[0], [clss[j], result[j]]), axis=0)
                encode_list.append(encode)

            encode_result.append(encode_list)
            print("Chosen classes: ", clss, result)
            print("Redimension   : ", np.shape(encode_list))
            # print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")

        return encode_result

    # TEST SEARCH 2
    def search_sample_2(self, sample):

        encode_result = []
        for i in range(len(sample)):
            x_ = [sample[i]]

            probability = self.sess.run(self.probVGG, feed_dict={self.x_batch: x_})
            probability = np.array(probability[0])

            clss = np.argsort(probability)[::-1]
            result = probability[clss]
            index = np.argwhere(result >= self.threshold).reshape(-1)

            # Clases elejidas y % de representacion
            clss = clss[index]
            result = result[index]

            # Normalizamos la data para los Auto-encoders
            assert np.shape(x_[0]) == np.shape(self.normalization_max), print('Different', np.shape(x_[0]), np.shape(self.normalization_max))
            x_ = x_/self.normalization_max

            # Insertamos la muestra original, la muestra original tiene como clase y probabilidad [-1,-1]
            decode_list = [np.concatenate((x_[0], [-1, -1]), axis=0)]
            for j in range(len(clss)):
                # Pasamos la muestra original por los AE de las clases elegidas
                cost_i, layer = self.sess.run([self.AEclass[clss[j]].cost, self.AEclass[j].y], feed_dict={self.x_batch: x_})
                # La muestra reconstruida, se concatena con su numero de clase y su valor de probabilidad
                decode = np.concatenate((layer[0], [clss[j], result[j]]), axis=0)
                decode_list.append(decode)

            # Con el AE global codificamos las muestras botenidas en el paso anterior
            encode_list = []
            for decode in decode_list:
                x_ = [decode[:-2]]
                label = decode[-2:]

                layer = self.sess.run(self.AEGlobal.z, feed_dict={self.x_batch: x_})
                encode = np.concatenate((layer[0], label), axis=0)
                encode_list.append(encode)

            encode_result.append(encode_list)
            print("Chosen classes: ", clss, result)
            print("Redimension   : ", np.shape(decode_list), '=>', np.shape(encode_list))
            print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")

        return encode_result

    # -----------------------------------------------------------------
    # Funciones secunadarias
    # -----------------------------------------------------------------
    # Layer FullConnected
    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
            return fc

    # Generate Parameter FullConnect layer
    def get_fc_var(self, in_size, out_size, name):
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
        weights = self.get_var_fc(initial_value, name, 'weights', name + "_weights")

        initial_value = tf.truncated_normal([out_size], .0, .001)
        biases = self.get_var_fc(initial_value, name, 'biases', name + "_biases")

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

