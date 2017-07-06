import sys, os
import numpy as np
from numpy import genfromtxt
from sklearn import neighbors, datasets, model_selection
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import PCA
import tensorflow as tf

switch_server = True
testdir = os.path.dirname('__file__')
srcdir = '..'
sys.path.insert(0, os.path.abspath(os.path.join(testdir, srcdir)))

# from onehotCNN.tune_subspace import getfractal
# from onehotCNN.tune_subspace import computelshparams

if switch_server is True:
    from tools import utils
    from nets import net_aencoder as AE
    from tools.dataset_csv import Dataset_csv
else:
    from tensorflow_manage_nets.tools import utils
    from tensorflow_manage_nets.nets import net_aencoder as AE
    from tensorflow_manage_nets.tools.dataset_csv import Dataset_csv

xpath = '../data/reduceDimension/'


def path_datasets(opc):
    if opc == 0:
        # MNIST
        data_name = 'MNIST'
        total = 10000
        path_data_test = [xpath + data_name + '/' + 'mnist-test-800.csv']
        path_data_train = [xpath + data_name + '/' + 'mnist-train-800.csv']
        path_max = xpath + data_name + '/' + 'max-mnist.csv'
        # dims = [63, 94, 141]
        #dims = [63, 94, 141, 211, 316]
        dims = [28, 42]
        origDim = 800
        method = 'pca'

    elif opc == 1:
        # CIFAR
        data_name = 'CIFAR10'
        total = 10000
        path_data_test = [xpath + data_name + '/' + 'cifar10-test-4096.csv']
        path_data_train = [xpath + data_name + '/' + 'cifar10-train-4096.csv']
        path_max = xpath + data_name + '/' + 'max-cifar10.csv'
        dims = [94, 141, 211, 316, 474]
        dims = [28, 42, 63]
        origDim = 4096
        method = 'pca'

    elif opc == 2:
        # SVHN
        data_name = 'SVHN'
        total = 26032
        path_data_test = [xpath + data_name + '/' + 'svhn-test-1152.csv']
        path_data_train = [xpath + data_name + '/' + 'svhn-train-1152.csv']
        path_max = xpath + data_name + '/' + 'max-svhn.csv'
        dims = [42, 63, 94, 141, 211]
        dims = [28, 42]
        origDim = 1152
        method = 'pca'

    elif opc == 3:
        # AGNews
        data_name = 'AGNEWS'
        total = 7552
        path_data_test = [xpath + data_name + '/' + 'agnews-test-8704.csv']
        path_data_train = [xpath + data_name + '/' + 'agnews-train-8704.csv']
        path_max = xpath + data_name + '/' + 'max-agnews.csv'
        dims = [94, 141, 211]
        # dims = [141, 211, 316]
        #dims = [42, 63,94,141]
        dims = [28]
        origDim = 8704
        method = 'inpca'

    return path_data_train, path_data_test, path_max, data_name, dims, method, origDim


def get_data_all(path_data, array_max):
    data_all = Dataset_csv(path_data=path_data, max_value=array_max)
    data_all.set_minibatch(data_all.total_inputs)
    X_data, y_data = data_all.generate_batch()

    return X_data, y_data, len(y_data)


# Función, fase de test
def test_model(net, sess_test, objData):
    total = objData.total_inputs
    cost_total = 0

    for i in range(objData.total_batchs_complete):
        x_, label = objData.generate_batch()
        cost = sess_test.run(net.cost, feed_dict={x_batch: x_})
        cost_total = cost_total + cost
        objData.next_batch_test()
    return cost_total, cost_total / total


def test_model_save(net, sess_test, objData, xdir, xname):
    total = objData.total_inputs
    cost_total = 0

    for i in range(objData.total_batchs_complete):
        x_, label = objData.generate_batch()
        cost, layer = sess_test.run([net.cost, net.z], feed_dict={x_batch: x_})
        utils.save_layer_output(layer, label, name=xname, dir=xdir)
        cost_total = cost_total + cost
        objData.next_batch_test()
    return cost_total, cost_total / total


def train_model(net, sess_train, objData, epoch):
    print('\n# PHASE: Training model')
    for ep in range(epoch):
        for i in range(objData.total_batchs):
            batch, _ = objData.generate_batch()
            sess_train.run(net.train, feed_dict={x_batch: batch})
            objData.next_batch()

        if ep % 5 == 0:
            cost_tot, cost_prom = test_model(net, sess_train, objData)
            print('     Epoch', ep, ': ', cost_tot, ' / ', cost_prom)


if __name__ == '__main__':

    epoch = 21
    learning_rate = 0.00008

    for opc in range(3,4):
        path_data_train_csv, path_data_test_csv, path_max_csv, name, dims, method, origDim = path_datasets(opc)
        Damax = utils.load_max_csvData(path_max_csv)

        data_train = Dataset_csv(path_data=path_data_train_csv, minibatch=35, max_value=Damax)
        # data_test = Dataset_csv(path_data=path_data_test_csv, minibatch=35, max_value=Damax, restrict=False)
        print('[', name, ']')

        for xdim in dims:
            print('     Dim:', xdim)

            pathFile = xpath + name + '/'

            with tf.Session() as sess:
                weight = xpath + name + '/' + 'weight-' + str(xdim) + '.npy'
                layers = [[int(origDim / 2), 'relu'], [xdim, 'relu']]

                x_batch = tf.placeholder(tf.float32, [None, origDim])
                ae = AE.AEncoder(weight, learning_rate=learning_rate)
                ae.build(x_batch, layers)
                sess.run(tf.global_variables_initializer())

                # TRAIN AENCODER
                train_model(ae, sess, data_train, epoch=epoch)
                ae.save_npy(sess, weight)

                # SAVE AENCODER
                # filenameTest = name.lower() + '-test-ae2-' + str(xdim)
                # filenameTrain = name.lower() + '-train-ae2-' + str(xdim)
                # cost_tot, cost_prom = test_model_save(ae, sess, data_train, pathFile, filenameTrain)
                # print('     TRAIN: Dim', xdim, ': ', cost_tot, ' / ', cost_prom)
                # cost_tot, cost_prom = test_model_save(ae, sess, data_test, pathFile, filenameTest)
                # print('     TEST : Dim', xdim, ': ', cost_tot, ' / ', cost_prom)
                # utils.normalization_complete([pathFile + 'output_' + filenameTest + '.csv', pathFile + 'output_' + filenameTrain + '.csv'])



