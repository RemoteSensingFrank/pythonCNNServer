#!/usr/bin/env python
#-*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np

#基础网络功能，包括：
#1.权重定义
#2.增益的定义
#3.二维卷积运算
#4.最大值池化
class BaseNet:
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

#LeNet网络的定义
#经典的深度网络LeNet，定义三层结构分别为L1层，L2层，全连接层，最后的输出
#L1,L2层包括卷积核，权重，偏置以及最大值池化
class LeNet(BaseNet):
    #定义第一层变量
    #输入参数为：1.卷积核大小；2.卷积核种类
    def layer1(self,corex,corey,corenumber):
        self.core1 = corenumber
        #第一层
        with tf.name_scope('layer1'):
            self.w1 = self.weight_variable([corex,corey,1,corenumber])#5×5的卷积核 32种特征
            self.b1 = self.bias_variable([32])
            self.h1 = tf.nn.relu(self.conv2d(self.x_image, self.w1) + self.b1)

            self.h_pool1 = self.max_pool_2x2(self.h1) #池化
            self.realxsize = self.realxsize/2
            self.realysize = self.realysize/2

            tf.summary.histogram('weight1',self.w1)
            tf.summary.histogram('bias1',self.b1)
    #定义第二层变量
    def layer2(self,corex,corey,corenumber):
        self.core2=corenumber
        #第二层
        with tf.name_scope('layer2'):
            self.w2 = self.weight_variable([corex,corey,self.core1,corenumber])
            self.b2 = self.bias_variable([64])
            self.h_conv2 = tf.nn.relu(self.conv2d(self.h_pool1, self.w2) + self.b2,name='layer2/relu')
            self.h_pool2 = self.max_pool_2x2(self.h_conv2)
            self.realxsize = self.realxsize/2
            self.realysize = self.realysize/2

            tf.summary.histogram('weight2',self.w2)
            tf.summary.histogram('bias2',self.b2)

    #定义全连接层
    def fullConnLayer(self,fulllayernumber):
        self.fullcore = fulllayernumber
        with tf.name_scope('full'):
            #全连接
            self.W_fc1 = self.weight_variable([int(self.realxsize*self.realysize*self.core2), int(fulllayernumber)])
            self.b_fc1 = self.bias_variable([fulllayernumber])
            self.h_pool2_flat = tf.reshape(self.h_pool2, [-1,int(self.realxsize*self.realysize*self.core2)])
            self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, self.W_fc1) + self.b_fc1,name='full/relu')

    #定义输出层
    def outputLayer(self,labellength):
        #输出
        #keep_prob=0.5
        #self.h_fc1_drop = tf.nn.dropout(self.h_fc1, keep_prob)
        with tf.name_scope('outlayer'):
            self.W_fc2 = self.weight_variable([self.fullcore, labellength])
            self.b_fc2 = self.bias_variable([labellength])
            self.y = tf.nn.softmax(tf.matmul(self.h_fc1, self.W_fc2) + self.b_fc2,name='outlayer/softmax')

    def __init__(self,xsize,ysize):
        self.realxsize = xsize
        self.realysize = ysize
                                              
        

