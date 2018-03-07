#!/usr/bin/env python
#-*- coding:utf-8 -*-
import tensorflow as tf


class Network:
    #初始化权重
    def weight_variable(self,shape):
        #从截断的正态分布中输出随机值
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    #初始化偏置
    def bias_variable(self,shape):
        #设置常数为0.1
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    #二维卷积运算
    def conv2d(self,x, W):
        return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="SAME")

    #最大值池化
    def max_pool_2x2(self,x):
        return tf.nn.max_pool(x, ksize=[1,2,2,1],
                              strides=[1,2,2,1], padding="SAME")

    def __init__(self):
        self.learning_rate = 0.001
        # 记录已经训练的次数
        self.global_step = tf.Variable(0, trainable=False)

        self.x = tf.placeholder(tf.float32, [None, 784])
        self.label = tf.placeholder(tf.float32, [None, 10])
        self.x_image = tf.reshape(self.x, [-1,28,28,1])

        #第一层
        self.w1 = self.weight_variable([5,5,1,32])#5×5的卷积核 32种特征
        self.b1 = self.bias_variable([32])
        self.h1 = tf.nn.relu(self.conv2d(self.x_image, self.w1) + self.b1)
        self.h_pool1 = self.max_pool_2x2(self.h1) #池化

        #第二层
        self.w2 = self.weight_variable([5,5,32,64])
        self.b2 = self.bias_variable([64])
        self.h_conv2 = tf.nn.relu(self.conv2d(self.h_pool1, self.w2) + self.b2)
        self.h_pool2 = self.max_pool_2x2(self.h_conv2)

        #全连接
        self.W_fc1 = self.weight_variable([7*7*64, 1024])
        self.b_fc1 = self.bias_variable([1024])
        self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, 7*7*64])
        self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, self.W_fc1) + self.b_fc1)

        #输出
        #keep_prob=0.5
        #self.h_fc1_drop = tf.nn.dropout(self.h_fc1, keep_prob)
        self.W_fc2 = self.weight_variable([1024, 10])
        self.b_fc2 = self.bias_variable([10])
        self.y = tf.nn.softmax(tf.matmul(self.h_fc1, self.W_fc2) + self.b_fc2)

        #loss
        self.loss = -tf.reduce_sum(self.label * tf.log(self.y + 1e-10))

        # minimize 可传入参数 global_step， 每次训练 global_step的值会增加1
        # 因此，可以通过计算self.global_step这个张量的值，知道当前训练了多少步
        self.train = tf.train.AdamOptimizer(self.learning_rate).minimize(
            self.loss, global_step=self.global_step)

        predict = tf.equal(tf.argmax(self.label, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(predict, "float"))
