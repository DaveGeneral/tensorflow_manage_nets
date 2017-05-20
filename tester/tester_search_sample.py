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
    from nets import net_complete_aencoder as AE
    from tools.dataset_csv import Dataset_csv
else:
    from tensorflow_manage_nets.tools import utils
    from tensorflow_manage_nets.nets import net_complete_aencoder as AE
    from tensorflow_manage_nets.tools.dataset_csv import Dataset_csv

# ..................................................................
# GLOBAL VARIABLES
dim_input = 4096
layers = [[1024, 'relu'], [256, 'relu']]
num_class = 10
capa = 'relu6'

# ..................................................................

# GLOBAL VARIABLES
path = '../data/features_cifar10_vgg/'
path_data_train_all = [path + 'output_trainVGG_'+capa+'.csv']
path_data_test_all = [path + 'output_testVGG_'+capa+'.csv']

path_weight = '../weight/vgg_cifar10/'
path_weight_ae = []

for i in range(num_class):
    path_weight_ae.append(path_weight + 'vggAE_class'+str(i)+'.npy')


assert os.path.exists(path), print('No existe el directorio de datos ' + path)
assert os.path.exists(path_weight), print('No existe el directorio de pesos ' + path_weight)

if __name__ == '__main__':

    # Datos de valor maximo
    #data_normal = Dataset_csv(path_data=[path_data_train_all[0], path_data_test_all[0]], random=False)
    #Damax = data_normal.amax
    #del data_normal

    # utils.generate_max_csvData([path_data_train_all[0], path_data_test_all[0]], path+'maximo.csv', has_label=True)
    Damax = utils.load_max_csvData(path+'maximo.csv')

    c = tf.ConfigProto()
    c.gpu_options.visible_device_list = "1,2"

    print('SEARCH SAMPLES')
    print('-----------------')

    data = Dataset_csv(path_data=path_data_test_all, minibatch=1, max_value=Damax, restrict=False, random=False)

    with tf.Session(config=c) as sess:
        aencoder = AE.ae_multiClass(session=sess, npy_weight_paths=path_weight_ae, num_class=num_class)
        aencoder.build(dim_input=dim_input, layers=layers)

        total = data.total_inputs
        cost_total = 0

        for i in range(data.total_batchs_complete):
            x, _ = data.generate_batch()

            res = aencoder.search_sample(sample=x)

            print(res)
            data.next_batch_test()



