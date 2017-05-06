import time
import tensorflow as tf
import sys, os
import numpy as np
import matplotlib.pyplot as plt

switch_server = True

testdir = os.path.dirname(__file__)
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


# GLOBAL VARIABLES
path = '../data/features/'
path_data_train = [path+'SKINfeaturesA_Test.csv']
path_data_test = [path+'SKINfeaturesA_Train.csv']
num_class = 2

# ..................................................................


# Función, fase de test
def test_model(net, sess_test, objData):

    total = objData.total_inputs
    count_success = 0
    count_by_class = np.zeros([num_class, num_class])
    prob_predicted = []

    # Iteraciones por Batch, en cada iteracion la session de tensorflow procesa los 'n' datos de entrada
    print('\n# PHASE: Test classification')
    for i in range(objData.total_batchs_complete):

        batch, label = objData.generate_batch()
        prob = sess_test.run(net.net['prob'], feed_dict={mlp_batch: batch, train_mode: False})

        # save output of a layer
        # utils.save_layer_output(layer, label, name='Train_SNC4_relu6')

        # Acumulamos la presicion de cada iteracion, para despues hacer un promedio
        count, count_by_class, prob_predicted = utils.print_accuracy(label, prob, matrix_confusion=count_by_class, predicted=prob_predicted)
        count_success = count_success + count
        objData.next_batch_test()

    # promediamos la presicion total
    accuracy_final = count_success/total
    print('\n# STATUS: Confusion Matrix')
    print(count_by_class)
    print('    Success total: ', str(count_success))
    print('    Accuracy total: ', str(accuracy_final))

    # a = objData.labels.tolist()
    # b = prob_predicted
    # cm = confusion_matrix(a, b)
    return accuracy_final


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
            label = tf.one_hot([li for li in label], on_value=1, off_value=0, depth=num_class)
            label = list(sess_train.run(label))
            # Run training
            t_start = time.time()
            _, cost = sess_train.run([net.train, net.cost], feed_dict={mlp_batch: batch, mlp_label: label, train_mode: True})
            t_end = time.time()
            # Next slice batch
            objData.next_batch()
            cost_i = cost_i + cost
            print("        > Minibatch: %d train on batch time: %7.3f seg." % (i, (t_end - t_start)))

        t1 = time.time()
        print("        Cost per epoch: ", cost_i)
        print("        Time epoch: %7.3f seg." % (t1 - t0))
        print("        Time per iteration: %7.3f seg." % ((t1 - t0) / epoch))


# ..................................................................


if __name__ == '__main__':

    path_load_weight = '../weight/saveMlpB_1.npy'
    path_save_weight = '../weight/saveMlpB_1.npy'

    mini_batch_train = 20
    mini_batch_test = 30
    epoch = 10
    learning_rate = 0.00005

    # GENERATE DATA
    # Datos de media y valor maximo
    data_normal = Dataset_csv(path_data=[path_data_train[0], path_data_test[0]], random=False)
    Damax = data_normal.amax
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
        accuracy = test_model(MLP, sess_test=sess, objData=data_test)

        # # SAVE WEIGHTs
        MLP.save_npy(sess, path_save_weight)
