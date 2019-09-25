import tensorflow as tf
import functools
import numpy as np

from surprise.common.tf_util import get_session, save_variables, load_variables
from surprise.common.tf_util import initialize
from surprise.ppo2.constants import constants

try:
    from surprise.common.mpi_adam_optimizer import MpiAdamOptimizer
    from mpi4py import MPI
    from surprise.common.mpi_util import sync_from_root
except ImportError:
    MPI = None

class Model(object):
    """
    We use this object to :
    __init__:
    - Creates the step_model
    - Creates the train_model

    train():
    - Make the training part (feedforward and retropropagation of gradients)

    save/load():
    - Save load the model
    """
    def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train,
                nsteps, ent_coef, vf_coef, max_grad_norm, mpi_rank_weight=1, comm=None,
                microbatch_size=None, predictor=None, model_type='SURPRISE',
                kl_div_w=0.01, rec_loss_w=0.99, inv_loss_w=0.8):

        self.sess = sess = get_session()
        self.predictor = predictor
        self.ac_space = ac_space
        print('nbatch_act',nbatch_act)
        print('nbatch_train',nbatch_train)
        print('microbatch_size',microbatch_size)

        if MPI is not None and comm is None:
            comm = MPI.COMM_WORLD

        with tf.variable_scope('ppo2_model', reuse=tf.AUTO_REUSE):
            # CREATE OUR TWO MODELS
            # act_model that is used for sampling
            act_model = policy(nbatch_act, 1, sess)

            # Train model for training
            if microbatch_size is None:
                train_model = policy(nbatch_train, nsteps, sess)
            else:
                train_model = policy(microbatch_size, nsteps, sess)

        # CREATE THE PLACEHOLDERS
        self.A = A = train_model.pdtype.sample_placeholder([None])
        self.ADV = ADV = tf.placeholder(tf.float32, [None])
        self.R = R = tf.placeholder(tf.float32, [None])
        #Adding attributes for extrinsic and intrinsic rewards
        self.EXTR_RWD = extr_rwds = tf.placeholder(tf.float32, [None])
        self.INTR_RWD = intr_rwds = tf.placeholder(tf.float32, [None])

        # Keep track of old actor
        self.OLDNEGLOGPAC = OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        # Keep track of old critic
        self.OLDVPRED = OLDVPRED = tf.placeholder(tf.float32, [None])
        self.LR = LR = tf.placeholder(tf.float32, [])
        # Cliprange
        self.CLIPRANGE = CLIPRANGE = tf.placeholder(tf.float32, [])

        neglogpac = train_model.pd.neglogp(A)

        # Calculate the entropy
        # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
        entropy = tf.reduce_mean(train_model.pd.entropy())

        # CALCULATE THE LOSS
        # Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss

        # Clip the value to reduce variability during Critic training
        # Get the predicted value
        vpred = train_model.vf
        vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        # Unclipped value
        vf_losses1 = tf.square(vpred - R)
        # Clipped value
        vf_losses2 = tf.square(vpredclipped - R)

        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

        # Calculate ratio (pi current policy / pi old policy)
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)

        # Defining Loss = - J is equivalent to max J
        pg_losses = -ADV * ratio

        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)

        # Final PG loss
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))

        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef

        #Renamed "predictor" to "perception", does not encapsulate inverse model but O.K.
        perception_loss = constants['PREDICTION_LR_SCALE'] * (self.predictor.invloss * (1-constants['FORWARD_LOSS_WT']) + self.predictor.forwardloss * constants['FORWARD_LOSS_WT'])

        if(model_type=='SURPRISE'):
            perception_loss = constants['PREDICTION_LR_SCALE'] * (inv_loss_w * self.predictor.invloss +
                tf.reduce_mean(kl_div_w * self.predictor.kld + rec_loss_w * self.predictor.rec_loss))

        # UPDATE THE PARAMETERS USING LOSS
        # 1. Get the model parameters
        policy_params = tf.trainable_variables('ppo2_model')
        # 2. Build our trainer
        if comm is not None and comm.Get_size() > 1:
            self.trainer = MpiAdamOptimizer(comm, learning_rate=LR, mpi_rank_weight=mpi_rank_weight, epsilon=1e-5)
        else:
            self.trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        # 3. Calculate the gradients
        policy_grads_and_var = self.trainer.compute_gradients(loss, policy_params)
        grads, var = zip(*policy_grads_and_var)

        # may need to factor out the loss as the original code
        # Original:  predgrads = tf.gradients(self.predloss * batchsize, predictor.var_list)
        # Factored out to make hyperparams not depend on it.

        #replaced loss by loss*batchsize (20, as was used in a3c.py)
        perception_grads_and_var = self.trainer.compute_gradients(perception_loss*20.0, self.predictor.var_list)
        perception_grads, perception_var = zip(*perception_grads_and_var)

        if max_grad_norm is not None:
            # Clip the gradients (normalize)
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
            # TODO: check if we need to change max_grad_norm for predictor gradients clipping
            # TODO: this still needs to be determined.
            perception_grads, _perception_grad_norm = tf.clip_by_global_norm(perception_grads, max_grad_norm)

        # zip aggregate each gradient with parameters associated
        # For instance zip(ABCD, xyza) => Ax, By, Cz, Da
        grads_and_var = list(zip(grads, var))
        perception_grads_and_var = list(zip(perception_grads, perception_var))

        all_grads_and_vars = grads_and_var + perception_grads_and_var

        #Added perception loss into stats
        self.grads = grads + perception_grads
        self.var = var + perception_var

        self._train_op = self.trainer.apply_gradients(all_grads_and_vars)
        self.predict_curiosity_bonus = self.predictor.pred_bonus

        #curiosity_bonus = self.predict_curiosity_bonus(self.predictor.s1[:], self.predictor.s2[:], self.predictor.asample)

        print_ret = tf.reduce_mean(R)
        print_extr_rwd = tf.reduce_mean(extr_rwds)
        print_intr_rwd = tf.reduce_mean(intr_rwds)

        if(model_type=='SURPRISE'):

            print_prior_p = tf.reduce_mean(self.predictor.prior_p)
            print_post_p = tf.reduce_mean(self.predictor.post_p)
            print_rec_loss = tf.reduce_mean(self.predictor.rec_loss)
            print_kld = tf.reduce_mean(self.predictor.kld)


        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac', 'perception_loss', 'inverse model loss', 'forward model loss', 'full returns', 'extrinsic rewards', 'intrinsic rewards']
        self.stats_list = [pg_loss, vf_loss, entropy, approxkl, clipfrac, perception_loss,
                        constants['PREDICTION_LR_SCALE'] * (self.predictor.invloss),
                        constants['PREDICTION_LR_SCALE'] * (self.predictor.forwardloss), print_ret, print_extr_rwd, print_intr_rwd]

        if(model_type=='SURPRISE'):

            self.loss_names.extend(['prior_p', 'post_p', 'rec_loss', 'kl_div'])
            self.stats_list.extend([ print_prior_p, print_post_p, print_rec_loss, print_kld])

        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.value = act_model.value
        self.initial_state = act_model.initial_state

        self.save = functools.partial(save_variables, sess=sess)
        self.load = functools.partial(load_variables, sess=sess)

        initialize()
        global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="")
        if MPI is not None:
            sync_from_root(sess, global_variables, comm=comm) #pylint: disable=E1101

    def train(self, lr, cliprange, obs, obs_after, returns,extr_rwds, intr_rwds, masks, actions, values, neglogpacs, states=None):
        # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
        # Returns = R + yV(s')
        advs = returns - values
        # Normalize the advantages
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        #NOTE: Commenting the below out. Uncomment if you want to see a stream of returns, extrinsinc and intrinsic rewards
        # print('returns / batch',np.mean(returns))
        # print('extrinsic rewards / batch ',np.mean(extr_rwds))
        # print('intrinsic rewards / batch ',np.mean(intr_rwds))

        td_map = {
            self.train_model.X : obs,
            self.A : actions,
            self.ADV : advs,
            self.R : returns,
            self.LR : lr,
            self.CLIPRANGE : cliprange,
            self.OLDNEGLOGPAC : neglogpacs,
            self.OLDVPRED : values,
            self.EXTR_RWD : extr_rwds,
            self.INTR_RWD : intr_rwds,
            # fetch directory for curiosity
            self.predictor.s1 : obs,
            self.predictor.s2 : obs_after,
            self.predictor.asample : self.actions_to_labels(actions),

            #Feed Dict of extrinsic and intrinsic rewards


        }
        if states is not None:
            td_map[self.train_model.S] = states
            td_map[self.train_model.M] = masks

        return self.sess.run(
            self.stats_list + [self._train_op],
            td_map
        )[:-1]


    # Takes into action with shape [batchsize] into [batch_size, action_space]
    # assume action_space is discrete here
    def actions_to_labels(self, actions):
        #print('actions', actions)
        #print('actions.shape[0]',actions.shape[0])
        #print('self.ac_space.n',self.ac_space.n)
        action_labels = np.zeros((actions.shape[0], self.ac_space.n))
        row_indexes = np.arange(actions.shape[0])
        column_indexes = actions
        action_labels[row_indexes, column_indexes] = 1.0
        #print(action_labels)
        return action_labels
