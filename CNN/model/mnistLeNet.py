#!/usr/bin/env python
#-*- coding:utf-8 -*-

import numpy as np
import tensorflow as tf
from PIL import Image
from model import LeNet
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../data/MNIST_data/",one_hot=True)

ckptfiles = './mnist_LeNet_ckpt/'

#mnist数据集构建LeNet网络
class mnistLeNet(LeNet):
    #初始化网络
    def __init__(self):
        LeNet.__init__(self,28,28)

        #控制变量定义
        self.learning_rate = 0.001
        # 记录已经训练的次数
        self.global_step = tf.Variable(0, trainable=False)
        self.x = tf.placeholder(tf.float32, [None, 784])
        self.label = tf.placeholder(tf.float32, [None, 10])
        self.x_image = tf.reshape(self.x, [-1,28,28,1])
        
        #网路层次定义
        self.layer1(5,5,32)
        self.layer2(5,5,64)
        self.fullConnLayer(int(1024))
        self.outputLayer(10)

        #计算参数定义
        #loss
        self.loss = -tf.reduce_sum(self.label * tf.log(self.y + 1e-10))

        # minimize 可传入参数 global_step， 每次训练 global_step的值会增加1
        # 因此，可以通过计算self.global_step这个张量的值，知道当前训练了多少步
        self.train = tf.train.AdamOptimizer(self.learning_rate).minimize(
            self.loss, global_step=self.global_step)

        predict = tf.equal(tf.argmax(self.label, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(predict, "float"))

#进行训练
class TrainMnistLeNet:
    def __init__(self):
        self.net = mnistLeNet()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.data = mnist

    #模型训练过程
    def trainMnist(self):
        batch_size = 50
        train_step = 3000
        # 记录训练次数, 初始化为0
        step = 0
        save_interval = 1000

        batch_size = 50
        train_step = 3000
        # 记录训练次数, 初始化为0
        step = 0

        # 每隔1000步保存模型
        #save_interval = 1000

        # tf.train.Saver是用来保存训练结果的。
        # max_to_keep 用来设置最多保存多少个模型，默认是5
        # 如果保存的模型超过这个值，最旧的模型将被删除
        saver = tf.train.Saver(max_to_keep=10)
        ckpt  = tf.train.get_checkpoint_state(ckptfiles)
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(ckptfiles+'graph',self.sess.graph)

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(self.sess, ckpt.model_checkpoint_path)
            # 读取网络中的global_step的值，即当前已经训练的次数
            step = self.sess.run(self.global_step)
            print('Continue from')
            print('        -> Minibatch update : ', step)

        while step < train_step:
            x, label = self.data.train.next_batch(batch_size)
            _, loss = self.sess.run([self.net.train, self.net.loss],
                                    feed_dict={self.net.x: x, self.net.label: label})
            step = self.sess.run(self.net.global_step)
            rs=self.sess.run(merged)
            writer.add_summary(rs, step)
            if step % 100 == 0:
                print('第%5d步，当前loss：%.2f' % (step, loss))

        # 模型保存在ckpt文件夹下
        # 模型文件名最后会增加global_step的值，比如1000的模型文件名为 model-1000
        #if step % save_interval == 0:
        #只保存一次模型
        saver.save(self.sess, ckptfiles+'model', global_step=step)

    #计算精度
    def calculate_accuracy(self):
        test_x = self.data.test.images
        test_label = self.data.test.labels
        accuracy = self.sess.run(self.net.accuracy,
                                 feed_dict={self.net.x: test_x, self.net.label: test_label})
        print("准确率: %.2f，共测试了%d张图片 " % (accuracy, len(test_label)))

#预测
class LeNetModel_io:
    #模型的初始化
    def __init__(self):
        self.net = mnistLeNet()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        #self.data = mnist
        # 加载模型到sess中
        self.restore(ckptfiles)

    #模型的读取
    def restore(self,path):
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            return "未保存任何模型"

    #读取图片转换为输入格式，然后进行预测
    def predict(self, image_path):
        # 读图片并转为黑白的
        img = Image.open(image_path).convert('F')
        im_resize = img.resize((28,28))
        flatten_img = np.reshape(im_resize, 784)
        flatten_img=flatten_img/255.0
        x = np.array([1 - flatten_img])
        y = self.sess.run(self.net.y, feed_dict={self.net.x: x})
        return np.argmax(y[0])

    #预测mnist数据集
    def predictMnist(self):
        x, label = mnist.train.next_batch(100)
        y = self.sess.run(self.net.y, feed_dict={self.net.x: x})
        for i in range(0,99):
            print(np.argmax(y[i]),label[i])

    def retrain(self,image_path,label):
        img = Image.open(image_path).convert('F')
        im_resize = img.resize((28,28))
        flatten_img = np.reshape(im_resize, 784)
        flatten_img=flatten_img/255.0
        x = np.array([1 - flatten_img])
        _, loss = self.sess.run([self.net.train, self.net.loss],
                                feed_dict={self.net.x: x, self.net.label: label})
        #只保存一次模型
        saver.save(self.sess, ckptfiles+'model', global_step=self.net.global_step+1)

"""
app = LeNetModel_io() 
print (app.predict('./ImageFiles/2018-03-06_23-32-38.png'))
print (app.predict('./ImageFiles/2018-03-06_23-44-23.png'))
"""
if __name__ == "__main__":
    app = TrainMnistLeNet()
    app.trainMnist()
    app.calculate_accuracy()