import sys, os
import numpy as np
import pylab as pl
from numpy import genfromtxt
from sklearn import neighbors, datasets, model_selection
import tensorflow as tf

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

xpath = '../data/reduceDimension/'


def path_datasets(opc):
    if opc == 0:
        # MNIST
        data_name = 'MNIST'
        total = 10000
        path_data_test = [xpath + data_name + '/' + 'mnist-test-800.csv']
        path_data_train = [xpath + data_name + '/' + 'mnist-train-800.csv']
        path_max = xpath + data_name + '/' + 'max-mnist.csv'
        dims = [9, 19, 28, 42, 63, 94, 141, 211]
        dims = [42, 94, 211]
        method = 'pca'


    elif opc == 1:
        # CIFAR
        data_name = 'CIFAR10'
        total = 10000
        path_data_test = [xpath + data_name + '/' + 'cifar10-test-4096.csv']
        path_data_train = [xpath + data_name + '/' + 'cifar10-train-4096.csv']
        path_max = xpath + data_name + '/' + 'max-cifar10.csv'
        dims = [9, 19, 28, 42, 63, 94, 141, 211]
        dims = [63, 141, 316]
        dims = [94, 211]
        method = 'pca'

    elif opc == 2:
        # SVHN
        data_name = 'SVHN'
        total = 26032
        path_data_test = [xpath + data_name + '/' + 'svhn-test-1152.csv']
        path_data_train = [xpath + data_name + '/' + 'svhn-train-1152.csv']
        path_max = xpath + data_name + '/' + 'max-svhn.csv'
        dims = [9, 19, 28, 42, 63, 94, 141, 211]
        dims = [28, 63, 141]
        method = 'pca'

    elif opc == 3:
        # AGNews
        data_name = 'AGNEWS'
        total = 7552
        path_data_test = [xpath + data_name + '/' + 'agnews-test-8704.csv']
        path_data_train = [xpath + data_name + '/' + 'agnews-train-8704.csv']
        path_max = xpath + data_name + '/' + 'max-agnews.csv'
        dims = [28, 42, 63, 94, 141, 211]
        dims = [63, 141, 316]
        method = 'pca'

    return path_data_train, path_data_test, path_max, data_name, dims, method


def get_data_split(path_data, array_max, test_size=0.3):
    data_all = Dataset_csv(path_data=path_data, max_value=array_max)
    data_all.set_minibatch(data_all.total_inputs)
    data, label = data_all.generate_batch()
    print(np.shape(data))

    X_train, X_test, y_train, y_test = model_selection.train_test_split(data, label, test_size=test_size,
                                                                        random_state=42)
    return X_train, X_test, y_train, y_test, len(y_train), len(y_test)


def get_data(path_data):
    matrix = genfromtxt(path_data, delimiter=',')
    shape = np.shape(matrix)
    X_data = matrix[:, :shape[1] - 1]
    y_data = np.ravel(matrix[:, -1:])
    total_data = len(y_data)
    print(np.shape(X_data), np.shape(y_data))

    return X_data, y_data, total_data


if __name__ == '__main__':

    path_logs = xpath + 'resultClassifier_PCA_11-07_CIFAR10.csv'
    f = open(path_logs, 'a')

    for i in range(1, 2):
        path_data_train_csv, path_data_test_csv, path_max_csv, name, dims, method = path_datasets(i)

        print('\n[NAME:', name, ']')
        for xdim in dims:
            path_reduce_train = xpath + name + '/' + name.lower() + '-train-' + method + '-' + str(xdim) + '-norm.csv'
            path_reduce_test = xpath + name + '/' + name.lower() + '-test-' + method + '-' + str(xdim) + '-norm.csv'
            print('     Dim:', xdim)
            print('     ', path_reduce_train)
            print('     ', path_reduce_test)

            # X_train, X_test, y_train, y_test, total_train, total_test = get_data_split([path_reduce_test], 1, 0.3)

            knn = neighbors.KNeighborsClassifier()
            print("     Train model...")
            X_train, y_train, total_train = get_data(path_reduce_train)
            knn.fit(X_train, y_train)
            print("     Test model...")
            X_test, y_test, total_test = get_data(path_reduce_test)
            Z = knn.predict(X_test)
            acc = utils.metrics_multiclass(y_test, Z)

            print('     Save result...')
            output = [name, total_test, acc, path_reduce_test, xdim, method]
            f.write(','.join(map(str, output)) + '\n')
            # f.write(','.join(map(str, y_test)) + '\n')
            # f.write(','.join(map(str, Z)) + '\n')
    f.close()

