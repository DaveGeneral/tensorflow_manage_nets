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

# ..................................................................

# GLOBAL VARIABLES
# opc = 2

pathx = '../../sharedfolder/DAsH/ann-benchmark/dataonehotcnn/'
path_weight = '../weight/onehotCNN/'
path_data_test = ''


def path_datasets(opc):
    if opc == 0:
        # DATA MNIST
        dim_input = 800
        name = 'mnist'
        path_data_train = [pathx + 'mnist-128-pca.csv']

    elif opc == 1:

        # DATA agNews
        dim_input = 8704
        name = 'agNews'
        path_data_train = [pathx+ 'agnews-256-pca.csv']

    elif opc == 2:

        # Data CIFAR
        dim_input = 4096
        name = 'cifar10'
        path_data_train = [pathx + 'cifar10-256-pca.csv']

    elif opc == 3:

        # DATA SVHN
        dim_input = 1152
        name = 'SVHN'
        path_data_train = [pathx + 'svhn-64-pca.csv']

    elif opc == 4:

        # DATA ISBI
        dim_input = 4096
        name = 'ISBI'
        path_data_train = [pathx + 'isbi-32-pca.csv']

    return name, path_data_train


# assert os.path.exists(path), print('No existe el directorio de datos ' + path)
# assert os.path.exists(path_weight), print('No existe el directorio de pesos ' + path_weight)

if __name__ == '__main__':

    directory = pathx + 'normalized/'

    for i in range(5):
        opc = i
        name, path_csv = path_datasets(opc)
        new_norm_csv = path_csv[0].split('/')[-1] + '_norm' + path_csv[0][-4:]

        print('Normalize Data', name, ':')
        utils.generate_MinMax_csvData(path_csv, pathx, name, has_label=True)
        utils.normalization_with_minMax([path_csv[0], pathx+'minimo_'+name+'.csv', pathx+'maximo_'+name+'.csv'], directory + new_norm_csv)
        print('complete!')

    print('Finish!!')

