import tensorflow as tf
import time
import os
import utils
from utils import VOC_dataset
import matplotlib.pyplot as plt
import numpy as np

#########################################################################
#Model:https://qiita.com/tktktks10/items/0f551aea27d2f62ef708
#########################################################################

cwd = os.getcwd()

class U_Net(object):

    def __init__(self):
        self.n_epoch = 1
        self.IMG_SIZE = 572
        self.TEACHER_IMG_SIZE = 388
        self.batch_size = 1
        self.NUM_CLASSES = 22

    def Main(self, inputs):

        #DOWNSAMPLING

        conv1 = utils.conv_2d(inputs, kernel_size=3, num_output=64, step=1, padding='VALID', scope='conv1')
        conv2 = utils.conv_2d(conv1, kernel_size=3, num_output=64, step=1, padding='VALID', scope='conv2')
        max_pool3 = utils.max_pool(conv2, pool_size=2, step=2, padding='SAME', scope='pool1')

        conv4 = utils.conv_2d(max_pool3, kernel_size=3, num_output=128, step=1, padding='VALID', scope='conv4')
        conv5 = utils.conv_2d(conv4, kernel_size=3, num_output=128, step=1, padding='VALID', scope='conv5')
        max_pool6 = utils.max_pool(conv5, pool_size=2, step=2, padding='SAME', scope='pool2')

        conv7 = utils.conv_2d(max_pool6, kernel_size=3, num_output=256, step=1, padding='VALID', scope='conv7')
        conv8 = utils.conv_2d(conv7, kernel_size=3, num_output=256, step=1, padding='VALID', scope='conv8')
        max_pool9 = utils.max_pool(conv8, pool_size=2, step=2, padding='SAME', scope='pool3')

        conv10 = utils.conv_2d(max_pool9, kernel_size=3, num_output=512, step=1, padding='VALID', scope='conv10')
        conv11 = utils.conv_2d(conv10, kernel_size=3, num_output=512, step=1, padding='VALID', scope='conv11')
        max_pool12 = utils.max_pool(conv11, pool_size=2, step=2, padding='SAME', scope='pool4')

        conv13 = utils.conv_2d(max_pool12, kernel_size=3, num_output=1024, step=1, padding='VALID', scope='conv13')
        conv14 = utils.conv_2d(conv13, kernel_size=3, num_output=1024, step=1, padding='VALID', scope='conv14')

        #START UPSAMPLING
        deconv1_shape = conv14.get_shape().as_list()[1]
        resize1_shape = deconv1_shape * 2
        deconv1 = utils.conv2_tp(conv14, kernel_size=2, num_output=512, step=2, padding='SAME', scope='deconv1')
        conv11_resize = tf.image.resize_image_with_crop_or_pad(conv11, resize1_shape, resize1_shape)
        concated1 = tf.concat([deconv1, conv11_resize], axis=3, name='concated1')
        conv15 = utils.conv_2d(concated1, kernel_size=3, num_output=512, step=1, padding='VALID', scope='conv15')
        conv16 = utils.conv_2d(conv15, kernel_size=3, num_output=512, step=1, padding='VALID', scope='conv16')

        deconv2_shape = conv16.get_shape().as_list()[1]
        resize2_shape = deconv2_shape * 2
        deconv2 = utils.conv2_tp(conv16, kernel_size=2, num_output=256, step=2, padding='SAME', scope='deconv2')
        conv8_resize = tf.image.resize_image_with_crop_or_pad(conv8, resize2_shape, resize2_shape)
        concated2 = tf.concat([deconv2, conv8_resize], axis=3, name='concated2')
        conv17 = utils.conv_2d(concated2, kernel_size=3, num_output=256, step=1, padding='VALID', scope='conv17')
        conv18 = utils.conv_2d(conv17, kernel_size=3, num_output=256, step=1, padding='VALID', scope='conv18')

        deconv3_shape = conv18.get_shape().as_list()[1]
        resize3_shape = deconv3_shape * 2
        deconv3 = utils.conv2_tp(conv18, kernel_size=2, num_output=128, step=2, padding='SAME', scope='deconv3')
        conv5_resize = tf.image.resize_image_with_crop_or_pad(conv5, resize3_shape, resize3_shape)
        concated3 = tf.concat([deconv3, conv5_resize], axis=3, name='concated3')
        conv19 = utils.conv_2d(concated3, kernel_size=3, num_output=128, step=1, padding='VALID', scope='conv19')
        conv20 = utils.conv_2d(conv19, kernel_size=3, num_output=128, step=1, padding='VALID', scope='conv20')

        deconv4_shape = conv20.get_shape().as_list()[1]
        resize4_shape = deconv4_shape * 2
        deconv4 = utils.conv2_tp(conv20, kernel_size=2, num_output=64, step=2, padding='SAME', scope='deconv4')
        conv2_resize = tf.image.resize_image_with_crop_or_pad(conv2, resize4_shape, resize4_shape)
        concated4 = tf.concat([deconv4, conv2_resize], axis=3, name='concated4')
        conv21 = utils.conv_2d(concated4, kernel_size=3, num_output=64, step=1, padding='VALID', scope='conv21')
        conv22 = utils.conv_2d(conv21, kernel_size=3, num_output=64, step=1, padding='VALID', scope='conv22')

        outputs = utils.conv_2d(conv22, kernel_size=1, num_output=self.NUM_CLASSES, step=1, activation=None, padding='SAME', scope='conv23')

        annotation_pred = tf.argmax(outputs, axis=3, name='prediction')

        return outputs, annotation_pred

    def Train(self):

        images = tf.placeholder(tf.float32, shape=[self.batch_size, self.IMG_SIZE, self.IMG_SIZE, 3], name="images")
        annotation = tf.placeholder(tf.float32, shape=[self.batch_size, self.TEACHER_IMG_SIZE, self.TEACHER_IMG_SIZE, 22], name="annotation")
        output, annotation_pred = self.Main(images)
        cost = utils.entropy_cost(logits=output, label=annotation)
        training_op = utils.training(cost)

        saver = tf.train.Saver()

        min_loss = None
        with tf.Session() as sess:
            print("==============================start train:{}==============================".format(time.ctime()))
            init = tf.global_variables_initializer()
            sess.run(init)

            train_ori_imgs, train_seg_imgs = VOC_dataset().next_batch(self.batch_size)

            for epoch in range(self.n_epoch):
                loss_train, _ = sess.run([cost, training_op], feed_dict={images:train_ori_imgs,
                                                                         annotation:train_seg_imgs})

                if min_loss is None:
                    min_loss = loss_train
                    saver.save(sess, cwd + "//U_NET/my_model.ckpt")
                elif loss_train < min_loss:
                    min_loss = loss_train
                    saver.save(sess, cwd + "//U_NET/my_model.ckpt")

                if epoch % 100 == 0:
                    print("step", epoch, "time", time, "loss:", loss_train)
            now = time.ctime()
            print("==============================end train:{}==============================".format(time.ctime()))


    def Test(self):

        model = tf.train.import_meta_graph('./U_NET/my_model.ckpt.meta')
        graph = tf.get_default_graph()
        images = graph.get_tensor_by_name('images:0')
        annotation_pred = graph.get_tensor_by_name('prediction:0')

        with tf.Session() as sess:
            model.restore(sess, './U_NET/my_model.ckpt')
            test_ori_imgs, _ = VOC_dataset().next_batch(self.batch_size)
            preds = sess.run([annotation_pred], feed_dict={images:test_ori_imgs})

            for pred in preds:
                plt.imshow(pred.reshape(388,388), cmap=plt.cm.binary)
                plt.show()

if __name__ =='__main__':
    #U_Net().Train()
    U_Net().Test()
