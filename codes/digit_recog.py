# -*-  coding : utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
import os
import numpy as np

# old_v = tf.logging.get_verbosity()
# tf.logging.set_verbosity(tf.logging.ERROR)

max_steps = 20000  # maximum steps
learning_rate = 0.0001  # learning rate
dropout = 0.9  # dropout rate
data_dir = './MNIST_DATA'
log_dir = './MNIST_LOG'

# acquire dataset and coding with one_hot
mnist = input_data.read_data_sets(data_dir, one_hot=True)

tf.reset_default_graph()

config = tf.ConfigProto(allow_soft_placement=True)
sess = tf.InteractiveSession(config=config)

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

# save image information
with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', image_shaped_input, 10)


# initialize weights
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# initialize bias
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# draw changes of parameters
def variable_summaries(var):
    with tf.name_scope('summaries'):
        # calculate mean
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)

        # calculate standard deviation
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        # use scalar to record
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))


# construct a CNN
def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    # set name space
    with tf.name_scope(layer_name):
        # initialize weight and record it
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
            variable_summaries(weights)
        # the same as weight
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            variable_summaries(biases)
        # linear compute
        with tf.name_scope('linear_compute'):
            preactivate = tf.matmul(input_tensor, weights) + biases
        # linear output with linear activation function
        activations = act(preactivate, name='activation')
    # output
    return activations


hidden1 = nn_layer(x, 784, 500, 'layer1')

# dropout layer
with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    tf.summary.scalar('dropout_keep_probability', keep_prob)
    dropped = tf.nn.dropout(hidden1, keep_prob)

y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)

# loss function
with tf.name_scope('loss'):
    # calculate cross_entropy
    diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
    with tf.name_scope('total'):
    #with tf.name_scope('total'):
        cross_entropy = tf.reduce_mean(diff)
    tf.summary.scalar('loss', cross_entropy)

# minimize cross entropy
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

# calculate accuarcy
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        # The index with the maximum value is extracted from the predicted and real tags respectively
        # the index with weak identities returns 1 (true), while the index with different values returns 0 (false).
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
        # the mean is accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()
# save results
train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
test_writer = tf.summary.FileWriter(log_dir + '/test')

# initialize all parameters
tf.global_variables_initializer().run()


def feed_dict(train):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    if train:
        xs, ys = mnist.train.next_batch(100)
        k = dropout
    else:
        xs, ys = mnist.test.images, mnist.test.labels
        k = 1.0
    return {x: xs, y_: ys, keep_prob: k}


for i in range(max_steps):
    if i % 100 == 0:
        summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
        test_writer.add_summary(summary, i)
        print('Accuracy at step %s: %s' % (i, acc))
    else:
        summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
        train_writer.add_summary(summary, i)

train_writer.close()
test_writer.close()

saver = tf.train.Saver()

n_epochs = 400
batch_size = 50

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(cross_entropy)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples//batch_size):
            x_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={x:x_batch,y_:y_batch, keep_prob: 1.0})
        acc_train=accuracy.eval(feed_dict={x:x_batch,y_:y_batch, keep_prob: 1.0})
        acc_test=accuracy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels, keep_prob: 1.0})
        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
    save_path = saver.save(sess, "./hwrecogmodel.ckpt")
    tf.train.write_graph(sess.graph_def, '', 'hwrecogmodel.pb')
