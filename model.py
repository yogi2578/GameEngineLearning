import tensorflow as tf
import numpy as np
import re
import sys
from Utils.tf_ops import ReLu, Conv2D, FC, Deconv2D

class Model:
    def __init__(self, s,a,x):
        """
        :param s: state- current stack of 4 image frames
        :param a: action- corresponding 4 actions
        :param x: next_state- next generated game image
        """
        self.s = s
        self.a = a
        self.x = x
        self.create_model()
        self.create_loss()
        self.create_optimizer()

    def create_model(self):
        """
        encode the 4 image stack, create action embeddings, use both in decoder to generate the next image
        """
        self.encode = self.create_encoder()
        self.act_embed = self.create_action_embedding()
        self.decode = self.create_decoder()

    def create_encoder(self):
        """
        encodes the 4 image stack
        """
        l = Conv2D(self.s, [6, 6], 64, 2, 'VALID', 'conv1')
        l = ReLu(l, 'relu1')
        l = Conv2D(l, [6, 6], 64, 2, 'SAME', 'conv2')
        l = ReLu(l, 'relu2')
        l = Conv2D(l, [6, 6], 64, 2, 'SAME', 'conv3')
        l = ReLu(l, 'relu3')
        l = FC(l, 1024, 'ip1')
        l = ReLu(l, 'relu4')
        l = FC(l, 2048, 'enc-factor', initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0))
        return l

    def create_action_embedding(self):
        """
        creates action embeddings
        """
        act = tf.cast(self.a, tf.float32)
        l = FC(act, 2048, 'act-embed', initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
        return l

    def create_decoder(self):
        """
        generates the next image
        """
        batch_size = tf.shape(self.encode)[0]
        l = tf.multiply(self.encode, self.act_embed, name='merge')
        l = FC(l, 1024, 'dec')
        l = FC(l, 64 * 10 * 10, 'ip4')
        l = ReLu(l, 'relu1')
        l = tf.reshape(l, [-1, 10, 10, 64], name='dec-reshape')
        l = Deconv2D(l, [6, 6], [batch_size, 20, 20, 64], 64, 2, 'SAME', 'deconv3')
        l = ReLu(l, 'relu2')
        l = Deconv2D(l, [6, 6], [batch_size, 40, 40, 64], 64, 2, 'SAME', 'deconv2')
        l = ReLu(l, 'relu3')
        l = Deconv2D(l, [6, 6], [batch_size, 84, 84, 3], 3, 2, 'VALID', 'x_hat-05')
        return l


    def create_loss(self):
        """
        loss function with regularization
        """
        lamb = 0.0001  # regularization
        penalty = tf.reduce_sum(lamb * tf.stack([tf.nn.l2_loss(var) for var in tf.trainable_variables()]),
                                name='regularization')
        self.loss = tf.reduce_mean(tf.nn.l2_loss(self.decode - self.x, name='l2') + penalty)
        #tensorboard
        tf.summary.scalar('loss',self.loss)

    def create_optimizer(self):
        """
        adam optimizer with learning rate decay and gradient clipping
        """
        lr = 1e-4
        global_step = tf.train.create_global_step()
        learning_rate = tf.train.exponential_decay(lr, global_step, 1e5, 0.9, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='optimizer')
        grads_vars = optimizer.compute_gradients(self.loss)
        bias_pattern = re.compile('.*/b')
        grads_vars_mult = []
        for grad, var in grads_vars:
            if bias_pattern.match(var.op.name):
                grads_vars_mult.append((grad * 2.0, var))
            else:
                grads_vars_mult.append((grad, var))
        grads_clip = [(tf.clip_by_value(grad, -0.1, 0.1), var) for grad, var in grads_vars_mult]
        self.step = optimizer.apply_gradients(grads_clip, global_step=global_step)