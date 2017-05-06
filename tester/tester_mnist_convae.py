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
    from nets import net_conv_aencoder as CAE
    from tools.dataset_csv import Dataset_csv
else:
    from tensorflow_manage_nets2.tools import utils
    from tensorflow_manage_nets2.nets import net_conv_aencoder as CAE
    from tensorflow_manage_nets2.tools.dataset_csv import Dataset_csv

# ..................................................................


# Plot example reconstructions
def plot_result(net, sess, batch, n_examples=5):
    print('Plotting')
    recon = sess.run(net.y, feed_dict={x_batch: batch})
    fig, axs = plt.subplots(2, n_examples, figsize=(n_examples, 2))
    for example_i in range(n_examples):
        axs[0][example_i].imshow(np.reshape(batch[example_i, :], (28, 28)))
        axs[1][example_i].imshow(np.reshape(recon[example_i, :], (28, 28)))

    fig.show()
    plt.draw()
    plt.waitforbuttonpress()
# ..................................................................

if __name__ == '__main__':

    path_load_weight = None
    path_save_weight = '../weight/MNIST_CAE3.npy'
    mini_batch_train = 20
    mini_batch_test = 25
    epoch = 10
    learning_rate = 0.0001
    noise_level = 0

    import tensorflow.examples.tutorials.mnist.input_data as input_data
    mnist = input_data.read_data_sets("../data/MNIST_data/", one_hot=True)
    trainX, testX = mnist.train, mnist.test

    with tf.Session() as sess:
        mean_img = np.mean(mnist.train.images, axis=0)
        x_batch = tf.placeholder(tf.float32, [None, 784])

        CAEncode = CAE.ConvAEncoder(npy_path=path_load_weight, learning_rate=learning_rate)
        CAEncode.build(input_batch=x_batch, n_filters=[1, 10, 10], corruption=False)
        sess.run(tf.global_variables_initializer())

        # Test inicial
        print('Original: ', sess.run(CAEncode.cost, feed_dict={x_batch: mnist.test.images}))

        # Fit all training data
        for epoch_i in range(epoch):
            for batch_i in range(mnist.train.num_examples // mini_batch_train):
                batch_xs, _ = trainX.next_batch(mini_batch_train)
                sess.run(CAEncode.train, feed_dict={x_batch: batch_xs})

            print(epoch_i, sess.run(CAEncode.cost, feed_dict={x_batch: mnist.test.images}))

        # Save
        CAEncode.save_npy(sess, path_save_weight)

        # Plot example reconstructions
        n_examples = 10
        plot_result(CAEncode, sess, mnist.test.next_batch(n_examples)[0], n_examples)
