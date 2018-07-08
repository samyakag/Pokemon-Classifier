import tensorflow.python.platform
from feature_format import featureFormat
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
# Global variables.
NUM_LABELS = 2    # The number of labels.
BATCH_SIZE = 100  # The number of training examples to use per training step.

# Define the flags useable from the command line.
tf.app.flags.DEFINE_string('train', None,
                           'File containing the training data (labels & features).')
tf.app.flags.DEFINE_string('test', None,
                           'File containing the test data (labels & features).')
tf.app.flags.DEFINE_integer('num_epochs', 1,
                            'Number of examples to separate from the training '
                            'data for the validation set.')
tf.app.flags.DEFINE_boolean('verbose', False, 'Produce verbose output.')
FLAGS = tf.app.flags.FLAGS
# print FLAGS.train
# print FLAGS.test
# Extract numpy representations of the labels and features given rows consisting of:
#   label, feat_0, feat_1, ..., feat_n


def main(argv=None):
    # Be verbose?
    verbose = FLAGS.verbose

    # Get the data.
    train_data_filename = FLAGS.train
    test_data_filename = FLAGS.test

    # Extract it into numpy matrices.
    # train_data, train_labels = featureFormat("Pokemon.csv")
    # test_data, test_labels = featureFormat("Pokemon.csv")
    features, labels, types = featureFormat("Pokemon.csv")
    # convert labels into one hot form
    enc = OneHotEncoder()
    enc.fit(labels)
    # labels = (np.arange(len(types)) == labels[:, None]).astype(np.float32)
    # Use one hot encoding for labels

    features, labels = shuffle(features, labels, random_state=1)
    train_features, test_features, test_labels, train_labels = train_test_split(features, labels, test_size=0.2, random_state=42)
    # Get the shape of the training data.
    # train_size, num_features = train_data.shape

    # Get the number of epochs for training.
    num_epochs = FLAGS.num_epochs

    # This is where training samples and labels are fed to the graph.
    # These placeholder nodes will be fed a batch of training data at each
    # training step using the {feed_dict} argument to the Run() call below.
    x = tf.placeholder("float", shape=[None, num_features])
    y_ = tf.placeholder("float", shape=[None, NUM_LABELS])

    # For the test data, hold the entire dataset in one constant node.
    test_data_node = tf.constant(test_data)

    # Define and initialize the network.

    # These are the weights that inform how much each feature contributes to
    # the classification.
    W = tf.Variable(tf.zeros([num_features, NUM_LABELS]))
    b = tf.Variable(tf.zeros([NUM_LABELS]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    # Optimization.
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    train_step = tf.train.GradientDescentOptimizer(
        0.01).minimize(cross_entropy)

    # Evaluation.
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # Create a local session to run this computation.
    with tf.Session() as s:
        # Run all the initializers to prepare the trainable parameters.
        tf.initialize_all_variables().run()
        if verbose:
            print 'Initialized!'
            print
            print 'Training.'

        # Iterate and train.
        for step in xrange(num_epochs * train_size // BATCH_SIZE):
            if verbose:
                print step,

            offset = (step * BATCH_SIZE) % train_size
            batch_data = train_data[offset:(offset + BATCH_SIZE), :]
            batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
            train_step.run(feed_dict={x: batch_data, y_: batch_labels})

            if verbose and offset >= train_size - BATCH_SIZE:
                print

        # Give very detailed output.
        if verbose:
            print
            print 'Weight matrix.'
            print s.run(W)
            print
            print 'Bias vector.'
            print s.run(b)
            print
            print "Applying model to first test instance."
            first = test_data[:1]
            print "Point =", first
            print "Wx+b = ", s.run(tf.matmul(first, W) + b)
            print "softmax(Wx+b) = ", s.run(tf.nn.softmax(tf.matmul(first, W) + b))
            print

        print "Accuracy:", accuracy.eval(feed_dict={x: test_data, y_: test_labels})


if __name__ == '__main__':
    tf.app.run()
