# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 11:19:35 2022
@author: mbp
"""
import numpy as np
#from functools import partial
#import matplotlib.pyplot as plt
#from scipy.io import loadmat, savemat
#from scipy.stats import kendalltau
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, Dropout, Add

class DeepSpeechPresence_ResNet(tf.keras.Model):
    def __init__(self, Q, K, K_size, structure, dropout=False, stoi_emulator=False):
        super(DeepSpeechPresence_ResNet, self).__init__()
        self.structure = structure
        self.L = int(np.size(structure) - np.sum(structure))
        self.M = int(np.sum(structure))
        self.res_blocks = []
        self.stoi_emulator = stoi_emulator
        
        def ResNet_block(K, K_size):
            block = []
            block.append(Conv2D(K, (K_size[0],K_size[1]), strides=(1,1), padding='same', activation=None, data_format='channels_last'))
            block.append(BatchNormalization(axis=3))
            block.append(LeakyReLU())
            block.append(Conv2D(K, (K_size[0],K_size[1]), strides=(1,1), padding='same', activation=None, data_format='channels_last'))
            if dropout:
                block.append(Dropout(0.25))
            block.append(BatchNormalization(axis=3))
            block.append(Add())
            block.append(LeakyReLU())
            return block
        
        def frequency_convolution(num):
            blocks = []
            for i in range(num):
                block = []
                block.append(Conv2D(Q, (1,Q), strides=(1,1), padding='valid', activation=None, data_format='channels_last'))
                block.append(Add())
                block.append(LeakyReLU())
                blocks.append(block)
            return blocks
            
        for l in range(self.L):
            self.res_blocks.append(ResNet_block(K, K_size))
        self.freq_convs = frequency_convolution(self.M)
        
        if self.stoi_emulator:
            self.conv_collapse = Conv2D(1, (1,Q), strides=(1,1), padding='valid', activation='sigmoid', data_format='channels_last')
        else:
            self.conv_collapse = Conv2D(Q, (1,Q), strides=(1,1), padding='valid', activation='sigmoid', data_format='channels_last')
#        self.conv_collapse = Conv2D(Q+1, (1,Q), strides=(1,1), padding='valid', activation='sigmoid', data_format='channels_last') # Q+1 kernels: the extra kernel is for VAD
        self.conv_collapse_activation = LeakyReLU()
        
    def call(self, inputs):
        def call_ResNet_block(x, block):
            X = block[0](x)
            for i in range(1,len(block)-2):
                X = block[i](X)
            X = block[-2]([X, x])
            X = block[-1](X)
            return X
        
        def call_freq_conv(x, block):
            X = block[0](x)
            X = tf.transpose(X, perm=(0,1,3,2))
            X = block[1]([X, x])
            X = block[2](X)
            return X
        
        l = 0
        m = 0
        
        x = inputs
        X = x

        for i in self.structure:
            if i == 0:
                X = call_ResNet_block(X, self.res_blocks[l])
                l += 1
            elif i == 1:
                X = call_freq_conv(X, self.freq_convs[m])
                m += 1
                
        X = self.conv_collapse(X)
        X = self.conv_collapse_activation(X)
        X = tf.transpose(X, perm=(0,1,3,2))
        if self.stoi_emulator:
            X = tf.math.reduce_mean(X, axis=(1,2))
            
        return X