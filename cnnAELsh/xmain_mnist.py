import time
import tensorflow as tf
import sys, os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from numpy import genfromtxt
import numpy as np

switch_server = True

testdir = os.path.dirname('__file__')
srcdir = '..'
sys.path.insert(0, os.path.abspath(os.path.join(testdir, srcdir)))

if switch_server is True:
    from tools import utils
    from cnnAELsh import net_cnnAELSH_mnist as CAL
    from tools.dataset_csv import Dataset_csv
else:
    from tensorflow_manage_nets.tools import utils
    from tensorflow_manage_nets.cnnAELsh import net_cnnAELSH_mnist as CAL
    from tensorflow_manage_nets.tools.dataset_csv import Dataset_csv

# ..................................................................
# GLOBAL VARIABLES
dim_input = 800
layers = [[400, 'relu'], [128, 'relu']]
num_class = 10

# ..................................................................

# DATA REDUCIDA
path = '../data/MNIST_dataA/'
path_data_train_all = [path + 'train800/output_train-mnist-800.csv']
path_data_test_all = [path + 'test800/output_test-mnist-800.csv']
path_normalization_max = path + 'maximo_mnist800.csv'

# PESOS ENTRENADOS
path_weight = '../weight/mnist/'
path_w_cnn = path_weight + 'save_cnn.npy'
path_w_ae_all = path_weight + 'save_ae_all.npy'
path_w_ae_class = []
for i in range(num_class):
    path_w_ae_class.append(path_weight + 'save_ae_class' + str(i) + '.npy')


def concatenate(out_layer, label):
    print("concatenate")
    total = len(label)
    lab = np.reshape(label, (total, 1))
    res = np.concatenate((out_layer, lab), axis=1)
    return res


def save_csv(X, file_name):
    print("saving ..." + file_name)
    f = open(file_name + '.csv', 'w')
    lenght = len(X)
    for i in range(lenght):
        f.write(','.join(map(str, X[i])) + '\n')
    f.close()


if __name__ == '__main__':
    c = tf.ConfigProto()
    c.gpu_options.visible_device_list = "1,2"

    print('CNN + AE + LSH')
    print('--------------')

    # data_train = Dataset_csv(path_data=path_data_train_all, minibatch=30, max_value=1, restrict=False, random=True)
    data = Dataset_csv(path_data=path_data_test_all, minibatch=30, max_value=1, restrict=False, random=False)

    with tf.device('/cpu:0'):
        with tf.Session() as sess:
            calsh = CAL.cnn_ae_lsh(session=sess,
                                   npy_convol_path=path_w_cnn,
                                   npy_ae_path=path_w_ae_all,
                                   npy_ae_class_paths=path_w_ae_class,
                                   normal_max_path=path_normalization_max,
                                   num_class=num_class,
                                   k_classes=1)

            calsh.build(dim_input=dim_input, layers=layers)

            #
            # TESTEAR PARTES DE LA RED
            # - - - - - - - - - - - -
            # Prueba la presicion de la CNN-VGG
            # calsh.test_vgg(data, normalize=False)
            # Prueba el error de reconstruccion del Autoencoder
            # calsh.test_ae_global(data, normalize=True)
            # Prueba de clasificacion con Autoencoders
            # calsh.test_ae_class(data, normalize=True)

            # Procces CNN+AE-Global-Class
            # Dim x = (1, 800) | return Dim (1, k_classes+1, 512+label+%=514)
            # x = []
            # result = calsh.search_sample(sample=x)

            X = genfromtxt(path_data_train_all[0], delimiter=',')
            shape = np.shape(X)
            print(shape)
            labelX = X[:, shape[1] - 1:]
            X = X[:, :shape[1] - 1]

            X = np.vstack(X)
            X = X.astype(np.float)

            # data = [[],[],[]]
            resultX = calsh.generate_data_encode_matrix(data=X, normalize=True)
            shape = np.shape(resultX)
            print(shape)
            # ------------------------------------
            Y = genfromtxt(path_data_test_all[0], delimiter=',')
            shape = np.shape(Y)
            print(shape)
            labelY = Y[:, shape[1] - 1:]
            Y = Y[:, :shape[1] - 1]

            Y = np.vstack(Y)
            Y = Y.astype(np.float)

            # data = [[],[],[]]
            resultY = calsh.generate_data_encode_matrix(data=Y, normalize=True)
            shape = np.shape(resultY)
            print(shape)

            # Save to CSV
            file_name_train = path + "get_csv128_train"
            file_name_test = path + "get_csv128_test"
            train = concatenate(resultX, labelX)
            test = concatenate(resultY, labelY)
            save_csv(train, file_name_train)
            save_csv(test, file_name_test)


