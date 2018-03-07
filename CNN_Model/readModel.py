#!/usr/bin/env python
#-*- coding:utf-8 -*-
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("../mnist/MNIST_data/",one_hot = True)
import os
import tensorflow as tf
import numpy as np
from PIL import Image
from cnnNetwork import Network
ckptfiles = '/home/wuwei/Program/Python/pythonCNNServer/CNN_Model/ckpt/'

'''
python 3.6
tensorflow 1.4
pillow(PIL) 4.3.0
使用tensorflow的模型来预测手写数字
输入是28 * 28像素的图片，输出是个具体的数字
'''
class Predict:
    def __init__(self):
        self.net = Network()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        #self.data = mnist
        # 加载模型到sess中
        self.restore()

    def restore(self):
        print ckptfiles
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(ckptfiles)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            return "未保存任何模型"

    def predict(self, image_path):
        # 读图片并转为黑白的
        img = Image.open(image_path).convert('F')
        im_resize = img.resize((28,28))
        flatten_img = np.reshape(im_resize, 784)
        flatten_img=flatten_img/255.0
        x = np.array([1 - flatten_img])
        y = self.sess.run(self.net.y, feed_dict={self.net.x: x})
        return np.argmax(y[0])

    def predictMnist(self):
        x, label = mnist.train.next_batch(100)
        y = self.sess.run(self.net.y, feed_dict={self.net.x: x})
        for i in range(0,99):
            print np.argmax(y[i]),label[i]

#app = Predict()
#print app.predict('/home/wuwei/Program/Python/CNNNunmberRecongnized/ImageFiles/2018-03-06_21-33-02.png')
#print app.predict('/home/wuwei/Program/Python/CNNNunmberRecongnized/ImageFiles/2018-03-06_21-32-55.png')
#print app.predict('/home/wuwei/Program/Python/CNNNunmberRecongnized/ImageFiles/2018-03-06_21-34-57.png')
#print app.predict('/home/wuwei/Program/Python/CNNNunmberRecongnized/ImageFiles/2018-03-06_21-35-08.png')
#print app.predict('/home/wuwei/Program/Python/CNNNunmberRecongnized/ImageFiles/2018-03-06_21-35-16.png')
#print app.predict('/home/wuwei/Program/Python/CNNNunmberRecongnized/ImageFiles/2018-03-06_21-35-24.png')
#print app.predict('/home/wuwei/Program/Python/CNNNunmberRecongnized/ImageFiles/2018-03-06_21-35-30.png')
