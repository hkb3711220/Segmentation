import tensorflow as tf
import os
import utils
import numpy as np
import matplotlib.pyplot as plt

##########################################################################
#Model: https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf
#Train_data:http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit
##########################################################################
os.chdir(os.path.dirname(__file__))

class FCN(object):

    def __init__(self):
        self.batch_size = 10
        self.n_epoch = 1

    def inference(self, images, pool, keep_prob, NUM_CLASSES):

        self.pool3 = pool[2]
        self.pool2 = pool[1]
        self.pool1 = pool[0]
        self.images = images

        bias6 = utils.create_bias(shape=[4096], scope='b6')
        conv6 = tf.nn.relu(utils.conv_2d(self.pool3, 6, 4096, step=1, scope='conv2d6') + bias6) # (1, 1, 4096)
        conv6_drop = tf.nn.dropout(conv6, keep_prob)

        bias7 = utils.create_bias(shape=[4096], scope='b7')
        conv7 = tf.nn.relu(utils.conv_2d(conv6_drop, 1, 4096, step=1, scope='conv2d7') + bias7) #(1, 1, 4096)
        conv7_drop = tf.nn.dropout(conv7, keep_prob)

        bias8 = utils.create_bias(shape=[NUM_CLASSES], scope='b8')
        conv8 = tf.nn.relu(utils.conv_2d(conv7_drop, 1, NUM_CLASSES, step=1, scope='conv2d8') + bias8) #(1, 1, 21)

        #BASE ON FCN-8
        #如下图所示，对原图像进行卷积conv1、pool1后原图像缩小为1/2；
        #之后对图像进行第二次conv2、pool2后图像缩小为1/4；
        #最后对图像进行第五次卷积操作conv5、pool5，缩小为原图像的1/8，
        #然后把原来CNN操作中的全连接变成卷积操作conv6、conv7，
        #图像的featureMap数量改变但是图像大小依然为原图的1/8，此时图像不再叫featureMap而是叫heatMap。

        #现在我们有1/8尺寸的heatMap，1/4尺寸的featureMap和1/2尺寸的featureMap，
        #1/8尺寸的heatMap进行upsampling操作之后，因为这样的操作还原的图片仅仅是conv5中的卷积核中的特征，
        #限于精度问题不能够很好地还原图像当中的特征，
        #因此在这里向前迭代。把conv4中的卷积核对上一次upsampling之后的图进行反卷积补充细节（相当于一个差值过程），
        #最后把conv3中的卷积核对刚才upsampling之后的图像进行再次反卷积补充细节，最后就完成了整个图像的还原。

        deconv1_shape = self.pool2.get_shape().as_list()
        print(deconv1_shape)
        bias9 = utils.create_bias(shape=[deconv1_shape[3]], scope='b9')
        deconv1 = utils.conv2_tp(conv8, 13, deconv1_shape[3], output_shape=tf.shape(self.pool2), scope='conv2_tp1') + bias9
        fuse1 = tf.add(self.pool2, deconv1, name="fuse_1")


        deconv2_shape = self.pool1.get_shape().as_list()
        bias10 = utils.create_bias(shape=[deconv2_shape[3]], scope='b10')
        deconv2 = utils.conv2_tp(fuse1, 3, deconv2_shape[3], output_shape=tf.shape(self.pool1), scope='conv2_tp2') + bias10
        fuse2 = tf.add(self.pool1, deconv2, name="fuse_2")

        shape = tf.shape(self.images)
        deconv3_shape = tf.stack([shape[0], shape[1], shape[2], NUM_CLASSES])
        bias11 = utils.create_bias(shape=[NUM_CLASSES], scope='b11')
        deconv3 = utils.conv2_tp(fuse2, 19, NUM_CLASSES, output_shape=deconv3_shape, step=8, scope='conv2_tp3') + bias11

        annotation_pred = tf.argmax(deconv3, axis=3, name="prediction")

        return annotation_pred, deconv3

    def train(self):
        model = tf.train.import_meta_graph('./my_model.ckpt.meta')
        graph = tf.get_default_graph()
        images = graph.get_tensor_by_name('images:0')
        annotation = tf.placeholder(tf.int32, shape=[None, 227, 227, 1], name="annotation")
        labels = tf.squeeze(annotation, axis=[3])
        print(labels)
        pool3 = graph.get_tensor_by_name('pool5/MaxPool:0')
        pool2 = graph.get_tensor_by_name('pool2/MaxPool:0')
        pool1 = graph.get_tensor_by_name('pool1/MaxPool:0')
        pool = [pool1, pool2, pool3]
        keep_prob = tf.placeholder_with_default(1.0, shape=(), name='keep_prob')

        annotation_pred, logits = self.inference(images, pool, keep_prob, NUM_CLASSES=21)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(self.n_epoch):
                images_batch, _ =  utils.next_batch(self.batch_size)
                prediction, logits = sess.run([annotation_pred, logits], feed_dict={images:images_batch})
                image = np.asarray(prediction[0])


    def FCN_32(self, conv):
        logitis = utils.conv

if __name__=='__main__':
    model = FCN()
    model.train()
