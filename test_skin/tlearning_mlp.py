import time
import tensorflow as tf
import sys, os
import numpy as np
from sklearn.preprocessing import label_binarize

switch_server = True

testdir = os.path.dirname('__file__')
srcdir = '..'
sys.path.insert(0, os.path.abspath(os.path.join(testdir, srcdir)))

if switch_server is True:
    from tools import utils
    from nets import net_mlperceptron as MLP
    from tools.dataset_csv import Dataset_csv
else:
    from tensorflow_manage_nets.tools import utils
    from tensorflow_manage_nets.nets import net_mlperceptron as MLP
    from tensorflow_manage_nets.tools.dataset_csv import Dataset_csv

# ..................................................................


# Función, fase de test
def test_model(net, sess_test, objData, plot_result=False):

    count_success = 0
    prob_predicted = []
    plot_predicted = []

    label_total = []
    prob_total = np.random.random((0, net.num_class))

    # Iteraciones por Batch, en cada iteracion la session de tensorflow procesa los 'n' datos de entrada
    print('\n# PHASE: Test classification')
    for i in range(objData.total_batchs_complete):

        batch, label = objData.generate_batch()
        prob = sess_test.run(net.net['prob'], feed_dict={mlp_batch: batch, train_mode: False})

        label_total = np.concatenate((label_total, label), axis=0)
        prob_total = np.concatenate((prob_total, prob), axis=0)

        # Acumulamos la presicion de cada iteracion, para despues hacer un promedio
        count, prob_predicted, plot_predicted = utils.process_prob(label, prob, predicted=prob_predicted, plot_predicted=plot_predicted)
        count_success = count_success + count
        objData.next_batch_test()

    # promediamos la presicion total
    print('\n# STATUS:')
    y_true = objData.labels
    y_prob = prob_predicted

    accuracy_total = utils.metrics(y_true, y_prob, plot_predicted, plot_result)
    # if plot_result is True:
    #     utils.precision_recall(y_true=label_total, y_prob=prob_total, num_class=net.num_class)

    return accuracy_total


# Funcion, fase de entrenamiento
def train_model(net, sess_train, objData, epoch):

    print('\n# PHASE: Training model')
    for ep in range(epoch):
        print('\n     Epoch:', ep)
        t0 = time.time()
        cost_i = 0
        for i in range(objData.total_batchs):
            batch, label = objData.generate_batch()
            # Generate the 'one hot' or labels
            label = label_binarize(label, classes=[i for i in range(net.num_class+1)])[:, :net.num_class]

            # Run training
            _, cost = sess_train.run([net.train, net.cost], feed_dict={mlp_batch: batch, mlp_label: label, train_mode: True})
            # Next slice batch
            objData.next_batch()
            cost_i = cost_i + cost

        t1 = time.time()
        print("        Cost per epoch: ", cost_i)
        print("        Time epoch: %7.3f seg." % (t1 - t0), " Time per iteration: %7.3f seg." % ((t1 - t0) / epoch))

# ..................................................................
#     Opcion para ejecutar la red con distintas fuentes de datos,
#                   son tres fuentes de datos

OPC = 'A'

# ..................................................................
path = '../data/features/muestra'+OPC+'/'
path_data_train = [path + 'SKINfeatures'+OPC+'_Train.csv']
path_data_test = [path + 'SKINfeatures'+OPC+'_Test.csv']
path_load_weight = None
path_save_weight = '../weight/tlmlp_'+OPC+'_1.npy'


if __name__ == '__main__':

    mini_batch_train = 20
    mini_batch_test = 30
    learning_rate = 0.0001
    epoch = 5
    num_class = 2

    # GENERATE DATA
    # Datos de media y valor maximo
    data_normal = Dataset_csv(path_data=[path_data_train[0], path_data_test[0]], random=False)
    Damax = data_normal.amax
    # utils.generate_max_csvData([path_data_train[0], path_data_test[0]], path+'maximo.csv', has_label=True)
    # Damax = utils.load_max_csvData(path+'maximo.csv')

    # Load data train
    data_train = Dataset_csv(path_data=path_data_train, minibatch=mini_batch_train, max_value=Damax, restrict=True)
    # Load data test
    data_test = Dataset_csv(path_data=path_data_test, minibatch=mini_batch_test, max_value=Damax, random=False)
    accuracy = 0

    with tf.Session() as sess:

        # DEFINE MODEL
        mlp_batch = tf.placeholder(tf.float32, [None, 4096])
        mlp_label = tf.placeholder(tf.float32, [None, num_class])
        train_mode = tf.placeholder(tf.bool)

        MLP = MLP.MLPerceptron(path_load_weight, learning_rate=learning_rate)
        MLP.build(mlp_batch, mlp_label, train_mode, layers=[2048, 1024, num_class])
        sess.run(tf.global_variables_initializer())

        test_model(MLP, sess_test=sess, objData=data_test)
        train_model(MLP, sess_train=sess, objData=data_train, epoch=epoch)
        accuracy = test_model(MLP, sess_test=sess, objData=data_test, plot_result=True)

        # # SAVE WEIGHTs
        MLP.save_npy(sess, path_save_weight)

    del MLP
