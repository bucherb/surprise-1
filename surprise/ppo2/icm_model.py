from __future__ import print_function
import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from surprise.ppo2.constants import constants


def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


def flatten(x):
    return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])


def conv2d(x, num_filters, name, filter_size=(3, 3), stride=(1, 1), pad="SAME", dtype=tf.float32, collections=None):
    with tf.variable_scope(name):
        stride_shape = [1, stride[0], stride[1], 1]
        filter_shape = [filter_size[0], filter_size[1], int(x.get_shape()[3]), num_filters]
        input_size = np.prod(filter_shape[:3])
        output_size = np.prod(filter_shape[:2]) * num_filters
        w_bound = np.sqrt(6. / (input_size + output_size))
        w = tf.get_variable("W", filter_shape, dtype, tf.random_uniform_initializer(-w_bound, w_bound), collections=collections)
        b = tf.get_variable("b", [1, 1, 1, num_filters], initializer=tf.constant_initializer(0.0), collections=collections)
        return tf.nn.conv2d(x, w, stride_shape, pad) + b

def linear(x, size, name, initializer=None, bias_init=0):
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=initializer)
    b = tf.get_variable(name + "/b", [size], initializer=tf.constant_initializer(bias_init))
    return tf.matmul(x, w) + b

def frame_encoder(x, nConvs=4):
    ''' 
        input: [None, 84, 84, 4];
        output: [None, 1152];
    '''
    print('Using universe head design')
    for i in range(nConvs):
        x = tf.nn.elu(conv2d(x, 32, "l{}".format(i + 1), [3, 3], [2, 2]))
    x = flatten(x)
    return x

class StateActionPredictor(object):
    def __init__(self, ob_space, ac_space):
        # input: s1,s2: : [None, h, w, ch] (usually ch=1 or 4)
        # asample: 1-hot encoding of sampled action from policy: [None, ac_space]
        input_shape = [None] + list(ob_space)
        self.s1 = phi1 = tf.placeholder(tf.float32, input_shape)
        self.s2 = phi2 = tf.placeholder(tf.float32, input_shape)
        self.asample = asample = tf.placeholder(tf.float32, [None, ac_space])

        # feature encoding: phi1, phi2: [None, LEN]
        size = 256
        phi1 = frame_encoder(phi1)
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            phi2 = frame_encoder(phi2)

        # inverse model: g(phi1,phi2) -> a_inv: [None, ac_space]
        g = tf.concat([phi1, phi2], 1)

        g = tf.nn.relu(linear(g, size, "g1", normalized_columns_initializer(0.01)))
        aindex = tf.argmax(asample, axis=1)  # aindex: [batch_size,]
        logits = linear(g, ac_space, "glast", normalized_columns_initializer(0.01))
        self.invloss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=aindex), name="invloss")
        self.ainvprobs = tf.nn.softmax(logits, dim=-1)

        # forward model: f(phi1,asample) -> phi2
        f = tf.concat([phi1, asample], 1)
        f = tf.nn.relu(linear(f, size, "f1", normalized_columns_initializer(0.01)))
        f = linear(f, phi1.get_shape()[1].value, "flast", normalized_columns_initializer(0.01))

        self.forward_loss_batch = 0.5 * tf.reduce_mean(tf.square(tf.subtract(f, phi2)), axis = 1, name='forwardloss')
        self.score = self.forward_loss_batch
        self.forwardloss = tf.reduce_mean(self.forward_loss_batch)

        # variable list
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def pred_act(self, s1, s2):
        sess = tf.get_default_session()
        return sess.run(self.ainvprobs, {self.s1: [s1], self.s2: [s2]})[0, :]

    def pred_bonus(self, s1, s2, asample):
        sess = tf.get_default_session()
        error = sess.run(self.score,
            {self.s1: s1, self.s2: s2, self.asample: asample})
        return error

