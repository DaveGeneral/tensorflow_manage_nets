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
        dims = [63]
        origDim = 800
        method = 'pca'

    elif opc == 1:
        # CIFAR
        data_name = 'CIFAR10'
        total = 10000
        path_data_test = [xpath + data_name + '/' + 'cifar10-test-4096.csv']
        path_data_train = [xpath + data_name + '/' + 'cifar10-train-4096.csv']
        path_max = xpath + data_name + '/' + 'max-cifar10.csv'
        dims = [94, 141, 211]
        origDim = 4096
        method = 'pca'

    elif opc == 2:
        # SVHN
        data_name = 'SVHN'
        total = 26032
        path_data_test = [xpath + data_name + '/' + 'svhn-test-1152.csv']
        path_data_train = [xpath + data_name + '/' + 'svhn-train-1152.csv']
        path_max = xpath + data_name + '/' + 'max-svhn.csv'
        dims = [42, 63, 94]
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
        origDim = 8704
        method = 'inpca'

    return path_data_train, path_data_test, path_max, data_name, dims, method, origDim


def get_data_all(path_data, array_max):
    data_all = Dataset_csv(path_data=path_data, max_value=array_max)
    data_all.set_minibatch(data_all.total_inputs)
    X_data, y_data = data_all.generate_batch()

    return X_data, y_data, len(y_data)

#
# def make_reduce_matrix(path_data, xmethod, dim_optimal, dataname, extraname):
#     matrix = genfromtxt(path_data, delimiter=',')
#     shape = np.shape(matrix)
#
#     X_data = matrix[:, :shape[1] - 1]
#     y_data = matrix[:, -1:]
#     total_data = len(y_data)
#
#     reducedMatrix = reduce_dimension_function(xmethod, X_data, dim_optimal)
#     filename_rm = xpath + dataname + '/' + dataname.lower() + '-' + extraname + '-' + xmethod + '-' + str(
#         dim_optimal) + '.csv'
#
#     print('filename_pca:', filename_rm)
#     f = open(filename_rm, "w")
#     for i in range(total_data):
#         f.write(",".join(map(str, np.concatenate((reducedMatrix[i], y_data[i]), axis=0))) + "\n")
#     f.close()
#     print('Save!!')
#
#     return filename_rm


# Función, fase de test
def test_model(net, sess_test, objData):

    total = objData.total_inputs
    cost_total = 0

    for i in range(objData.total_batchs_complete):

        x_, label = objData.generate_batch()
        cost = sess_test.run(net.cost, feed_dict={x_batch: x_})
        cost_total = cost_total + cost
        objData.next_batch_test()
    return cost_total, cost_total/total


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

    epoch = 15
    learning_rate = 0.0001

    for opc in range(0, 1):
        path_data_train_csv, path_data_test_csv, path_max_csv, name, dims, method, origDim = path_datasets(opc)
        Damax = utils.load_max_csvData(path_max_csv)

        data_train = Dataset_csv(path_data=path_data_train_csv, minibatch=50, max_value=Damax)
        #data_test = Dataset_csv(path_data=path_data_test_csv, minibatch=50, max_value=Damax, restrict=False, random=False)
        print('[', name, ']')

        for xdim in dims:

            with tf.Session() as sess:

                layers = [[int(origDim/2), 'relu'], [xdim, 'relu']]

                x_batch = tf.placeholder(tf.float32, [None, origDim])
                ae = AE.AEncoder(None, learning_rate=learning_rate)
                ae.build(x_batch, layers)
                sess.run(tf.global_variables_initializer())

                train_model(ae, sess, data_train, epoch=epoch)

            # a = make_reduce_matrix(path_data_test_csv[0], method, xdim, name, 'test')
            # b = make_reduce_matrix(path_data_train_csv[0], method, xdim, name, 'train')
            # utils.normalization_complete([a, b])




