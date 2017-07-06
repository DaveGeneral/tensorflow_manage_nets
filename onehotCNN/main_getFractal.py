import time
import tensorflow as tf
import sys, os
import numpy as np
import matplotlib.pyplot as plt
from tune_subspace import getfractal
from tune_subspace import computelshparams
from sklearn.metrics import confusion_matrix
from numpy import genfromtxt
import numpy as np

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

# ..................................................................

# GLOBAL VARIABLES
xpath = '../data/reduceDimension/'


def path_datasets(opc):
    if opc == 0:
        # MNIST
        data_name = 'MNIST'
        total = 10000
        path_data_test = [xpath + data_name + '/' + 'mnist-test-800.csv']
        path_data_train = [xpath + data_name + '/' + 'mnist-train-800.csv']
        path_max = xpath + data_name + '/' + 'max-mnist.csv'
        dims = [28, 42, 63, 94, 141, 211, 316]
        origDim = 800
        method = 'pca'

    elif opc == 1:
        # CIFAR
        data_name = 'CIFAR10'
        total = 10000
        path_data_test = [xpath + data_name + '/' + 'cifar10-test-4096.csv']
        path_data_train = [xpath + data_name + '/' + 'cifar10-train-4096.csv']
        path_max = xpath + data_name + '/' + 'max-cifar10.csv'
        dims = [28, 42, 63, 94, 141, 211, 316, 474]
        origDim = 4096
        method = 'pca'

    elif opc == 2:
        # SVHN
        data_name = 'SVHN'
        total = 26032
        path_data_test = [xpath + data_name + '/' + 'svhn-test-1152.csv']
        path_data_train = [xpath + data_name + '/' + 'svhn-train-1152.csv']
        path_max = xpath + data_name + '/' + 'max-svhn.csv'
        dims = [28, 42, 63, 94, 141, 211]
        origDim = 1152
        method = 'pca'

    elif opc == 3:
        # AGNews
        data_name = 'AGNEWS'
        total = 7552
        path_data_test = [xpath + data_name + '/' + 'agnews-test-8704.csv']
        path_data_train = [xpath + data_name + '/' + 'agnews-train-8704.csv']
        path_max = xpath + data_name + '/' + 'max-agnews.csv'
        dims = [28, 42, 63, 94, 141, 211]
        origDim = 8704
        method = 'inpca'

    return path_data_train, path_data_test, path_max, data_name, dims, method, origDim


if __name__ == '__main__':

    results = xpath + 'resultAE.function_fractal'
    f = open(results, 'a')
    for x in range(4, 5):
        opc = x
        path_data_train_csv, path_data_test_csv, path_max_csv, name, dims, method, origDim = path_datasets(opc)

        print("[ DATASET", name, ']')

        dimData = []
        dimFractal = []
        pathFile = xpath + name + '/'

        for xdim in dims:
            print('     '+name, 'Dim:', xdim)
            filenameTest = 'output_' + name.lower() + '-test-ae2-' + str(xdim) + '-norm.csv'
            # filenameTrain = 'output_' + name.lower() + '-train-ae2-' + str(xdim) + '-norm.csv'

            Xmatrix = genfromtxt(filenameTest, delimiter=',')
            shape = np.shape(Xmatrix)
            labels = Xmatrix[:, -1:]
            Xmatrix = Xmatrix[:, :shape[1] - 1]

            XFractal = getfractal(pathFile + 'dimFractal/', filenameTest, Xmatrix)
            print("     fractal dimension of dim-" + str(xdim) + ':', XFractal)

            dimData.append(xdim)
            dimFractal.append(XFractal)

        output = [name, 'ae2-norm']
        f.write(','.join(map(str, output)) + '\n')
        f.write(','.join(map(str, dimData)) + '\n')
        f.write(','.join(map(str, dimFractal)) + '\n')

    f.close()


