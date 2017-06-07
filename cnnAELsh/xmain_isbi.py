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
    from cnnAELsh import net_cnnAELSH_isbi as CAL
    from tools.dataset_csv import Dataset_csv
else:
    from tensorflow_manage_nets.tools import utils
    from tensorflow_manage_nets.cnnAELsh import net_cnnAELSH_isbi as CAL
    from tensorflow_manage_nets.tools.dataset_csv import Dataset_csv

# ..................................................................
# GLOBAL VARIABLES
dim_input = 4096
layers = [[2048, 'sigmoid'], [512, 'sigmoid']]
num_class = 2

# ..................................................................

# DATA REDUCIDA
path = '../data/features/testskin1/muestraA/'
path_data_train_all = [path + 'SKINfeaturesA_Train.csv']
path_data_test_all = [path + 'SKINfeaturesA_Test.csv']
path_normalization_max = path + 'maximo_ISBI.csv'

# PESOS ENTRENADOS
path_weight = '../weight/ISBI/'
path_w_cnn = path_weight + 'save_cnn.npy'
path_w_ae_all = path_weight + 'save_ae_all.npy'
path_w_ae_class = []
for i in range(num_class):
    path_w_ae_class.append(path_weight + 'save_ae_class'+str(i)+'.npy')


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
            # Dim x = (1, 4096) | return Dim (1, k_classes+1, 512+label+%=514)
            #x = []
            #result = calsh.search_sample_2(sample=x)

            #data = [[],[],[]]
            #calsh.generate_data_encode_matrix(data=data, normalize=True)

