import tensorflow as tf
import os
import cv2
import pandas as pd
import utils

########################################################
#https://blog.csdn.net/sunbaigui/article/details/39938097
#train_data:http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit
########################################################
cwd = os.getcwd()

class ALexNet(object):

    def __init__(self, NUM_CLASSES):
        self.batch_size = 10
        self.n_epoch = 1
        self.NUM_CLASSES = NUM_CLASSES

    def classifier(self, inputs, keep_prob, NUM_CLASSES):

        self.inputs = inputs
        bias1 = utils.create_bias(shape=[96], scope='b1')
        conv1 = tf.nn.relu(utils.conv_2d(self.inputs, 11, 96, step=4, scope='conv2d1') + bias1)
        pool1 = utils.max_pool(conv1, 3, 2, scope='pool1')
        lrn1 = tf.nn.local_response_normalization(pool1, name='lrn1')

        bias2 = utils.create_bias(shape=[256], scope='b2')
        lrn1_pad = tf.pad(lrn1, [[0,0], [2,2], [2,2], [0,0]])
        conv2 = tf.nn.relu(utils.conv_2d(lrn1_pad, 5, 256, step=1, scope='conv2d2') + bias2)
        pool2 = utils.max_pool(conv2, 3, 2, scope='pool2')
        lrn2 = tf.nn.local_response_normalization(pool2, name='lrn2')

        bias3 = utils.create_bias(shape=[384], scope='b3')
        lrn2_pad = tf.pad(lrn2, [[0,0], [1,1], [1,1], [0,0]])
        conv3 = tf.nn.relu(utils.conv_2d(lrn2_pad, 3, 384, step=1, scope='conv2d3') + bias3)

        bias4 = utils.create_bias(shape=[384], scope='b4')
        conv3_pad = tf.pad(conv3, [[0,0], [1,1], [1,1], [0,0]])
        conv4 = tf.nn.relu(utils.conv_2d(conv3_pad, 3, 384, step=1, scope='conv2d4') + bias4)

        bias5 = utils.create_bias(shape=[256], scope='b5')
        conv4_pad = tf.pad(conv4, [[0,0], [1,1], [1,1], [0,0]])
        conv5 = tf.nn.relu(utils.conv_2d(conv4_pad, 3, 256, step=1, scope='conv2d5') + bias5, name='conv5')
        pool5 = utils.max_pool(conv5, 3, 2, scope='pool5')

        pool5_flat = tf.reshape(pool5, [-1, 6*6*256]) #Flatten
        fc1 = utils.fully_connected(pool5_flat, 4096, scope='fc1')
        fc1_drop = tf.nn.dropout(fc1, keep_prob)

        fc2 = utils.fully_connected(fc1_drop, 4096, scope='fc2')
        fc2_drop = tf.nn.dropout(fc2, keep_prob)

        logits = utils.fully_connected(fc2_drop, NUM_CLASSES, activation_fn=None, scope='fc3')
        prediction = tf.nn.softmax(logits)

        return logits, prediction

    def train(self):

        images = tf.placeholder(tf.float32, shape=[None, 227, 227, 3], name='images')
        labels = tf.placeholder(tf.int32, shape=[None,])
        keep_prob = tf.placeholder_with_default(1.0, shape=(), name='keep_prob')
        logits, prediciton = self.classifier(images, keep_prob, self.NUM_CLASSES)
        cost = utils.entropy_cost(logits, labels)
        training_op = utils.training(cost)

        cost_summary = tf.summary.scalar('cost', cost)
        saver = tf.train.Saver()

        with tf.Session() as sess:
            min_loss = None
            sess.run(tf.global_variables_initializer())
            for epoch in range(self.n_epoch):
                images_batch, labels_batch =  utils.next_batch(self.batch_size)
                _, loss_train, _ = sess.run([training_op, cost, cost_summary], feed_dict={images:images_batch,
                                                                                          labels:labels_batch,
                                                                                          keep_prob:0.5})

                if min_loss is None:
                    min_loss = loss_train
                    saver.save(sess, cwd + "//my_model.ckpt")
                elif loss_train < min_loss:
                    min_loss = loss_train
                    saver.save(sess, cwd + "//my_model.ckpt")

                print("step", epoch, "loss:", loss_train)

        FileWriter = tf.summary.FileWriter("./log", graph=tf.get_default_graph())

if __name__=='__main__':
    model = ALexNet(4251)
    model.train()
