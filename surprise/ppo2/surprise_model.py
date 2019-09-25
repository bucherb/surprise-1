from __future__ import print_function
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.contrib.rnn as rnn


def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

def flatten(x):
    return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])

def frame_encoder(x, nConvs=4):
    ''' encoder network
        input: [None, 84, 84, 4];
        output: [None, 1152];
    '''
    for i in range(nConvs):
        name = "l{}".format(i + 1)
        input_size = np.prod([3,3,int(x.get_shape()[3])])
        output_size = np.prod([3,3]) * 32
        w_bound = np.sqrt(6. / (input_size + output_size))
        x = tf.nn.elu(tf.layers.conv2d(x, 32, kernel_size=[3,3], strides=[2,2], name=name,
            kernel_initializer=tf.random_uniform_initializer(-w_bound, w_bound),
            bias_initializer=tf.constant_initializer(0.0)))
    x = flatten(x)
    return x

def log_likelihood(mean,std,x):
    log_likelihood = (-0.5*tf.log(tf.square(std)) - 0.5*tf.square(x - mean)/tf.square(std))
    return tf.reduce_mean(log_likelihood, axis=1)

class StateActionPredictor(object):
    def __init__(self, ob_space, ac_space):
        # input: s1,s2: : [None, h, w, ch] (usually ch=1 or 4)
        # asample: 1-hot encoding of sampled action from policy: [None, ac_space]
        input_shape = [None] + list(ob_space)
        self.s1 = phi1 = tf.placeholder(tf.float32, input_shape)
        self.s2 = phi2 = tf.placeholder(tf.float32, input_shape)
        self.asample = asample = tf.placeholder(tf.float32, [None, ac_space])

        # latent dimension size
        latent_dim = 64

        # feature encoding: phi1, phi2: [None, LEN]
        size = 256
        phi1 = frame_encoder(phi1)
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            phi2 = frame_encoder(phi2)

        self.phi2_enc = phi2

        # inverse model: g(phi1,phi2) -> a_inv: [None, ac_space]
        g = tf.concat([phi1, phi2], 1)
        g = tf.nn.relu(tf.layers.dense(g, size, name="g1", kernel_initializer=normalized_columns_initializer(0.01)))
        aindex = tf.argmax(asample, axis=1)
        logits = tf.layers.dense(g, ac_space, name="glast", kernel_initializer=normalized_columns_initializer(0.01))

        self.invloss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=aindex), name="invloss")
        self.ainvprobs = tf.nn.softmax(logits, axis=-1)

        # CVAE forward model
        f = tf.concat([phi1, phi2, asample], 1)
        f = tf.nn.relu(tf.layers.dense(f, size, name="cvae_enc_l1", kernel_initializer=normalized_columns_initializer(0.01)))
        f = tf.layers.dense(f, latent_dim*2, name="cvae_enc_l2", kernel_initializer=normalized_columns_initializer(0.01))

        mean = f[:,:latent_dim]
        std = tf.nn.softplus(f[:,latent_dim:]) + 1e-6

        # sample from latent space
        z = mean + std*tf.random_normal(tf.shape(mean, out_type = tf.int32), 0, 1, dtype = tf.float32)
        self.z = z

        # decoder
        # phi1 - initial state
        # asample - action
        # z - sample from latent space
        dec = tf.concat([phi1,asample,z], 1)
        dec = tf.nn.relu(tf.layers.dense(dec, size, name="cvae_dec_l1", kernel_initializer=normalized_columns_initializer(0.01)))
        dec = tf.layers.dense(dec, phi2.get_shape()[1].value, name="cvae_dec_l2", kernel_initializer=normalized_columns_initializer(0.01))
        self.reconstructed = dec

        # reconstruction loss
        # phi2 - next encoded state - observed, tensor
        # dec - next encoded state - predicted, tensor
        # Mean Squared Error
        self.rec_loss = 0.5 * tf.reduce_mean(tf.square(tf.subtract(dec, phi2)), axis=1)

        # score calculation
        prior_mean = tf.fill(tf.shape(mean, out_type = tf.int32),0.0)
        prior_std = tf.fill(tf.shape(std, out_type = tf.int32),1.0)

        # KL divergence - between latent distribution and Gaussian with mean 0 and std 1
        target_dist = tfp.distributions.Normal(loc=prior_mean, scale=prior_std)
        current_dist = tfp.distributions.Normal(loc=mean, scale=std)
        KL_div = tf.reduce_mean(tfp.distributions.kl_divergence(current_dist, target_dist), axis=1)
        # scale the kl divergence by the constant beta factor for forward loss calculation
        self.kld = KL_div

        # prior normal distribution likelihood
        self.prior_p = log_likelihood(prior_mean, prior_std, z)
        # posterior distribution likelihood
        self.post_p = log_likelihood(mean,std,z)

        # score for the intrinsic reward calculation, for a single sample
        # score is the lowest like next state decision
        # p(z|x) -> self.prior_p is the likelihood of z under prior distribution, which is normal
        # q(z|x,y) -> self.post_p is the likelihood of z under the posterior distribution, which is the output of the encoder

        # -log(p)   take negative log of the marginal probability
        self.score = - 1.0 * (-self.rec_loss + self.prior_p - self.post_p)
        # forward loss: reconstruction loss + KL divergernce
        self.forwardloss = tf.reduce_mean(self.rec_loss + KL_div, name='forwardloss')

        # variable list
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def pred_act(self, s1, s2):
        sess = tf.get_default_session()
        return sess.run(self.ainvprobs, {self.s1: [s1], self.s2: [s2]})[0, :]

    def pred_bonus(self, s1, s2, asample):
        sess = tf.get_default_session()
        curiosity_reward = sess.run((self.score,self.z,self.phi2_enc,self.reconstructed,self.rec_loss,self.kld),
            {self.s1: s1, self.s2: s2, self.asample: asample})

        return curiosity_reward
