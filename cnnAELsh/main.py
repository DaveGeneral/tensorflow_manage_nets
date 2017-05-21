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
    from cnnAELsh import net_cnnAELSH as CAL
    from tools.dataset_csv import Dataset_csv
else:
    from tensorflow_manage_nets.tools import utils
    from tensorflow_manage_nets.cnnAELsh import net_cnnAELSH as CAL
    from tensorflow_manage_nets.tools.dataset_csv import Dataset_csv

# ..................................................................
# GLOBAL VARIABLES
dim_input = 4096
layers = [[2048,'relu'], [512,'relu']]
num_class = 10

# ..................................................................

# DATA REDUCIDA
path = '../data/features_cifar10_vgg/'
path_data_train_all = [path + 'output_trainVGG_relu6.csv']
path_data_test_all = [path + 'output_testVGG_relu6.csv']

# PESOS ENTRENADOS
path_weight = '../weight/vgg_cifar10/'
path_w_cnn = path_weight + 'save_vgg.npy'
path_w_ae_all = path_weight + 'save_ae_all.npy'
path_w_ae_class = []
for i in range(num_class):
    path_w_ae_class.append(path_weight + 'save_ae_class'+str(i)+'.npy')

assert os.path.exists(path), print('No existe el directorio de datos ' + path)
assert os.path.exists(path_weight), print('No existe el directorio de pesos ' + path_weight)

if __name__ == '__main__':

    # Datos de valor maximo
    # data_normal = Dataset_csv(path_data=[path_data_train_all[0], path_data_test_all[0]], random=False)
    # Damax = data_normal.amax
    # del data_normal

    # utils.generate_max_csvData([path_data_train_all[0], path_data_test_all[0]], path+'maximo.csv', has_label=True)
    Damax = utils.load_max_csvData(path + 'maximo.csv')

    c = tf.ConfigProto()
    c.gpu_options.visible_device_list = "1,2"

    print('SEARCH SAMPLES')
    print('--------------')

    data = Dataset_csv(path_data=path_data_test_all, minibatch=30, max_value=1, restrict=False, random=False)

    with tf.device('/cpu:0'):
        with tf.Session() as sess:
            calsh = CAL.cnn_ae_lsh(session=sess,
                                   npy_convol_path=path_w_cnn,
                                   npy_ae_path=path_w_ae_all,
                                   normal_max_path=path + 'maximo.csv',
                                   num_class=num_class)
            calsh.build(dim_input=dim_input, layers=layers)
            calsh.train_ae_global(data, normalizate=True)

            # for i in range(data.total_batchs_complete):
            #     x, label = data.generate_batch()
            #
            #     res = aencoder.search_sample(sample=x)
            #     data.next_batch_test()
            #     print(res, label)
