# coding = utf-8
import tensorflow as tf
from PIL import Image, ImageFilter
import os
import numpy as np

record_dir = 'num_record'
# if path does not exist, create one
if not os.path.exists(record_dir):
    os.makedirs(record_dir)

# the below procedure can also be completed with numpy
# get binary image
def binary_con():
    src_test = Image.open('./test5.bmp').convert('L')
    # resize the test image into 28*28
    src_test = src_test.resize((28,28),Image.ANTIALIAS)
    threshold = 110
    table = []
    for i in range(256):
        if i < threshold:
            table.append(0)
        else:
            table.append(1)
    src_test_middle = src_test.point(table, '1')
    src_test_middle = src_test_middle.filter(ImageFilter.EDGE_ENHANCE_MORE)
    return src_test_middle

def imageprepare():
    # save the image
    src_test_bin = binary_con()
    src_test_bin.save("binary_test.jpg")
    src_test_bin = list(src_test_bin.getdata())
    src_test_ = [(255 - x)*1.0/255.0 for x in src_test_bin]
    return src_test_

src_test_ = imageprepare()
learning_rate = 0.0001
dropout = 0.9

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

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # read the model
    saver.restore(sess, "./hwrecogmodel.ckpt")
    print("Model restored.")
    prediction=tf.argmax(y, 1)
    predic = prediction.eval(feed_dict={x: [src_test_],keep_prob: 1.0}, session=sess)
    print("result =", predic[0])
    # save the result,could also be done via numpy
    num_txt = record_dir + '.txt'
    record = open(num_txt, 'wb')
    record.write(predic[0])
    record.close()
    #np.savetxt(num_txt,predic.astype(int))
    print(open('./num_record.txt','r').read())
