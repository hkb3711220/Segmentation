import tensorflow as tf
import pandas as pd
import numpy as np
import os
import cv2
import pickle

os.chdir(os.path.dirname(__file__))
train = pd.read_csv('./train.csv')

def create_dict():

    if os.path.exists('./label.dump') == False:
        ids = set(list(train['Id']))
        label_dict = {}
        for i, label in enumerate(ids):
            label_dict[label] = i

        with open('label.dump', 'wb') as f:
            pickle.dump(label_dict, f)

    label_dict = pickle.load(open('label.dump', 'rb'))

    return label_dict


def img_read(img_name, classifier=False):
    if classifier:
        PATH = './Resized'
    else:
        PATH = './Pictures'

    img = cv2.imread(os.path.join(PATH, img_name), cv2.IMREAD_COLOR)

    return img

def next_batch(batch_size):

    imgs_name = list(train['Image'])
    labels = list(train['Id'])
    label_dict = create_dict()

    idx = np.arange(0 , len(imgs_name))
    np.random.shuffle(idx)
    idx = idx[:batch_size]

    img_name_shuffles = [imgs_name[i] for i in idx]
    label_shuffles = [label_dict[labels[i]] for i in idx]

    imgs_shuffles = []
    for img_name in img_name_shuffles:
        img = img_read(img_name, classifier=True)
        imgs_shuffles.append(img)

    return np.asarray(imgs_shuffles), np.asarray(label_shuffles)

def create_weight(shape, scope=None):
    weight = tf.truncated_normal(shape)
    return tf.Variable(weight)

def create_bias(shape, scope=None):
    bias = tf.constant(0.0, shape=shape, dtype=tf.float32)
    return tf.Variable(bias)

def conv_2d(inputs, kernel_size, num_output, step=1, padding='VALID',scope=None):
    with tf.variable_scope(scope):
        num_input = inputs.get_shape().as_list()[3]
        kernel = create_weight([kernel_size, kernel_size, num_input, num_output])
        stride = [1, step, step, 1]
        conv = tf.nn.conv2d(inputs, kernel, stride, padding=padding)

    return conv

def conv2_tp(inputs, kernel_size, num_output, output_shape=None, step=2, scope=None):

    with tf.variable_scope(scope):
        if output_shape is None:
            output_shape = inputs.get_shape().as_list()
            output_shape[1] = (output_shape[1] -1) * step + kernel_size
            output_shape[2] = (output_shape[2] -1) * step + kernel_size
            output_shape[3] = num_output

        inputs_shape = inputs.get_shape().as_list()
        num_input = inputs_shape[3]
        kernel = create_weight([kernel_size, kernel_size, num_output, num_input])

        #filter: A 4-D Tensor with the same type as value and shape [height, width, output_channels, in_channels]. filter's in_channels dimension must match that of value.

        deconv = tf.nn.conv2d_transpose(inputs,
                                        kernel,
                                        output_shape = output_shape,
                                        strides = [1, step, step, 1],
                                        padding="VALID")
    return deconv

def max_pool(inputs, pool_size, step, scope=None):
    with tf.variable_scope(scope):
        max_pool = tf.nn.max_pool(inputs,
                                  ksize=[1, pool_size, pool_size, 1], #Filter [2, 2]
                                  strides=[1, step, step, 1],
                                  padding='VALID')

    return max_pool

def fully_connected(inputs, NUM_CLASSES, activation_fn=tf.nn.relu ,scope=None):
    with tf.variable_scope(scope):
        fc = tf.contrib.layers.fully_connected(inputs,
                                               NUM_CLASSES,
                                               activation_fn=activation_fn,
                                               weights_initializer = tf.contrib.layers.xavier_initializer(),
                                               biases_initializer = tf.zeros_initializer())

    return fc

def entropy_cost(logits, label):
    crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logits)
    cost = tf.reduce_mean(crossent)

    return cost

def training(cost):
    max_gradient_norm = 1
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    params = tf.trainable_variables()
    gradients = tf.gradients(cost, params)
    clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
    training_op = optimizer.apply_gradients(zip(clipped_gradients, params))

    return training_op
