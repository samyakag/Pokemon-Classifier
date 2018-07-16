import tensorflow.python.platform
from feature_format import featureFormat
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
import sys
import matplotlib.pyplot as plt
np.set_printoptions(threshold=sys.maxint)


# Define the flags useable from the command line.
tf.app.flags.DEFINE_float('learningRate', None,
                          'File containing the training data (labels & features).')
tf.app.flags.DEFINE_integer('num_epochs', 1,
                            'Number of examples to separate from the training '
                            'data for the validation set.')
FLAGS = tf.app.flags.FLAGS
# print FLAGS.train
# print FLAGS.test
# Extract numpy representations of the labels and features given rows consisting of:
#   label, feat_0, feat_1, ..., feat_n


def main(argv=None):
    num_epochs = FLAGS.num_epochs
    learningRate = FLAGS.learningRate
    features, labels, types = featureFormat("Pokemon.csv")
    features, labels = shuffle(features, labels, random_state=1)
    # Split the data into testing and training sets
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=0.2, random_state=42)

    num_labels = len(types)
    num_features = features.shape[1]
    x = tf.placeholder("float", shape=[None, num_features])
    W = tf.Variable(tf.zeros([num_features, num_labels]))
    b = tf.Variable(tf.zeros([num_labels]))
    y_ = tf.placeholder("float", shape=[None, num_labels])

    weights = {
        'h1': tf.Variable(tf.truncated_normal([num_features, 60])),
        'h2': tf.Variable(tf.truncated_normal([60, 60])),
        'h3': tf.Variable(tf.truncated_normal([60, 60])),
        'h4': tf.Variable(tf.truncated_normal([60, 60])),
        'out': tf.Variable(tf.truncated_normal([60, num_labels]))
    }

    biases = {
        'b1': tf.Variable(tf.truncated_normal([60])),
        'b2': tf.Variable(tf.truncated_normal([60])),
        'b3': tf.Variable(tf.truncated_normal([60])),
        'b4': tf.Variable(tf.truncated_normal([60])),
        'out': tf.Variable(tf.truncated_normal([num_labels]))
    }

    model_path = "./model"
    init = tf.initialize_all_variables()

    saver = tf.train.Saver()

    y = multilayer_perceptron(x, weights, biases)
    # y = _generateTensorLayers(x, weights, biases)

    cost_function = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
    training_step = tf.train.GradientDescentOptimizer(
        learningRate).minimize(cost_function)

    sess = tf.Session()
    sess.run(init)
    mse_history = []
    accuracy_history = []
    cost_history = np.empty(shape=[1], dtype=float)

    for epoch in range(num_epochs):
        sess.run(training_step, feed_dict={
                 x: train_features, y_: train_labels})
        cost = sess.run(cost_function, feed_dict={
                        x: train_features, y_: train_labels})
        cost_history = np.append(cost_history, cost)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        # tf.Print(y, [y])
        # sess.run(y, feed_dict={x: train_features})
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        pred_y = sess.run(y, feed_dict={x: test_features})
        mse = tf.reduce_mean(tf.square(pred_y - test_labels))
        mse_history.append(sess.run(mse))
        accuracy = (sess.run(accuracy, feed_dict={
                    x: train_features, y_: train_labels}))
        accuracy_history.append(accuracy)

        print('epoch: ', epoch, ' - cost: ', cost,
              " - MSE: ", mse, "- Train Accuracy: ", accuracy)
    save_path = saver.save(sess, model_path)
    print(" Model Saved in file: {}".format(save_path))
    pred_y = sess.run(y, feed_dict={x: test_features})
    mse = tf.reduce_mean(tf.square(pred_y - test_labels))
    print("MSE : {}".format(sess.run(mse)))

    # plt.plot(mse_history, "r")
    # plt.show()
    # plt.plot(accuracy_history)
    # plt.show()

    # correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # print("Test Accuracy: ", (sess.run(accuracy, feed_dict={x: test_features, y_: test_labels})))

    # pred_y = sess.run(y, feed_dict={x: test_features})
    # mse = tf.reduce_mean(tf.square(pred_y - test_labels))
    # print("MSE: %.4f" % sess.run(mse))
    # print(test_labels.shape)
# Create a local session to run this computation.
    # with tf.Session() as s:
    #     # Run all the initializers to prepare the trainable parameters.
    #     tf.initialize_all_variables().run()
    #     if verbose:
    #         print 'Initialized!'
    #         # print
    #         print 'Training.'

    #     train_step.run(feed_dict={x: train_features, y_: train_labels})
    #     # Iterate and train.
    #     # for step in xrange(num_epochs * train_features.shape[0] // BATCH_SIZE):
    #     #     if verbose:
    #     #         print step,
    #     #         print
    #     #     offset = (step * BATCH_SIZE) % train_features.shape[0]
    #     #     batch_features = train_features[offset:(offset + BATCH_SIZE), :]
    #     #     batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
    #     #     train_step.run(feed_dict={x: batch_features, y_: batch_labels})
    #     #     # print(batch_features.shape)
    #     #     print(batch_features.shape)

    #     #     if verbose and offset >= train_features.shape[0] - BATCH_SIZE:
    #     #         print

    #     # Give very detailed output.
    #     if verbose:
    #         print
    #         print 'Weight matrix.'
    #         print s.run(W)
    #         print
    #         print 'Bias vector.'
    #         print s.run(b)
    #         print
    #         print "Applying model to first test instance."
    #         first = test_features[:1]
    #         print "Point =", first
    #         print "Wx+b = ", s.run(tf.matmul(first, W) + b)
    #         print "softmax(Wx+b) = ", s.run(tf.nn.softmax(tf.matmul(first, W) + b))
    #         print

    # print "Accuracy:", accuracy.eval(feed_dict={x: test_features, y_:
    # test_labels})


def multilayer_perceptron(x, weights, biases):

    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.sigmoid(layer_2)

    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.sigmoid(layer_3)

    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    layer_4 = tf.nn.relu(layer_4)

    out_layer = tf.matmul(layer_4, weights['out']) + biases['out']
    print "Hello"
    return out_layer


if __name__ == '__main__':
    tf.app.run()
