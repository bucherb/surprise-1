import os
import time
import numpy as np
import json
import os.path as osp
from surprise import logger
from collections import deque
from surprise.common import explained_variance, set_global_seeds
from surprise.common.policies import build_policy
from surprise.ppo2.runner import Runner


def constfn(val):
    def f(_):
        return val
    return f

def learn(*, network, env, total_timesteps, eval_env = None, seed=None, nsteps=2048, ent_coef=0.0, lr=3e-4,
            vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95, log_interval=10, nminibatches=4,
            noptepochs=4, cliprange=0.2, save_interval=10, model_type = 'SURPRISE',
            extrinsic_factor = 1.0, intrinsic_factor = 1.0, restore_global_checkpoint=True,
            kl_div_w=0.01, rec_loss_w=0.99, inv_loss_w=0.8,
            global_checkpoint_path="~/models/global_checkpoint/ckp000000", resume_file="/NAS/home/res_itr.txt", need_log_imgs=False, load_path=None,
            model_fn=None, update_fn=None, init_fn=None, mpi_rank_weight=1, comm=None, **network_kwargs):

    '''
    Learn policy using PPO algorithm (https://arxiv.org/abs/1707.06347)

    Parameters:
    ----------

    network:                          policy network architecture. Either string (mlp, lstm, lnlstm, cnn_lstm, cnn, cnn_small, conv_only - see surprise.common/models.py for full list)
                                      specifying the standard network architecture, or a function that takes tensorflow tensor as input and returns
                                      tuple (output_tensor, extra_feed) where output tensor is the last network layer output, extra_feed is None for feed-forward
                                      neural nets, and extra_feed is a dictionary describing how to feed state into the network for recurrent neural nets.
                                      See common/models.py/lstm for more details on using recurrent nets in policies

    env: surprise.common.vec_env.VecEnv     environment. Needs to be vectorized for parallel environment simulation.
                                      The environments produced by gym.make can be wrapped using surprise.common.vec_env.DummyVecEnv class.


    nsteps: int                       number of steps of the vectorized environment per update (i.e. batch size is nsteps * nenv where
                                      nenv is number of environment copies simulated in parallel)

    total_timesteps: int              number of timesteps (i.e. number of actions taken in the environment)

    ent_coef: float                   policy entropy coefficient in the optimization objective

    lr: float or function             learning rate, constant or a schedule function [0,1] -> R+ where 1 is beginning of the
                                      training and 0 is the end of the training.

    vf_coef: float                    value function loss coefficient in the optimization objective

    max_grad_norm: float or None      gradient norm clipping coefficient

    gamma: float                      discounting factor

    lam: float                        advantage estimation discounting factor (lambda in the paper)

    log_interval: int                 number of timesteps between logging events

    nminibatches: int                 number of training minibatches per update. For recurrent policies,
                                      should be smaller or equal than number of environments run in parallel.

    noptepochs: int                   number of training epochs per update

    cliprange: float or function      clipping range, constant or schedule function [0,1] -> R+ where 1 is beginning of the training
                                      and 0 is the end of the training

    save_interval: int                number of timesteps between saving events

    load_path: str                    path to load the model from

    model_type : str                  'SURPRISE' or 'ICM', default is SURPRISE

    extrinsic_factor: float            Default is 1.0, denotes whether to use extrinsic rewards and how scaled

    intrinsic_factor: float            Default is 1.0, denotes whether to use intrinsic rewards and how scaled

    **network_kwargs:                 keyword arguments to the policy / network builder. See surprise.common/policies.py/build_policy and arguments to a particular type of network
                                      For instance, 'mlp' network architecture has arguments num_hidden and num_layers.
    '''

    print('total_timesteps', total_timesteps)
    
    set_global_seeds(seed)

    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)
    total_timesteps = int(total_timesteps)

    policy = build_policy(env, network, **network_kwargs)

    # Get the nb of env
    nenvs = env.num_envs

    # Get state_space and action_space
    ob_space = env.observation_space
    ac_space = env.action_space

    # Calculate the batch_size
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches

    # Initialize the predictor
    if model_type == 'SURPRISE':
        # surprise model
        from surprise.ppo2.surprise_model import StateActionPredictor
    else:
        # ICM model
        from surprise.ppo2.icm_model import StateActionPredictor
    predictor = StateActionPredictor(ob_space.shape, ac_space.n)

    # Instantiate the model object (that creates act_model and train_model)
    if model_fn is None:
        from surprise.ppo2.model import Model
        model_fn = Model

    model = model_fn(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                    nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                    max_grad_norm=max_grad_norm, comm=comm, mpi_rank_weight=mpi_rank_weight, predictor=predictor,
                    model_type=model_type, kl_div_w=0.01, rec_loss_w=0.99, inv_loss_w=0.8)

    res_itr = 1
    global_checkpoint_path = osp.expanduser(global_checkpoint_path)
    if load_path is not None:
            model.load(load_path)
            print('----loaded the model from load_path: ' + load_path + '----')

    # Automatically restores the model from global_checkpoint_path by default
    # unless the restore_global_checkpoint arg is False
    elif restore_global_checkpoint:
        if (os.path.isfile(global_checkpoint_path)):
            model.load(global_checkpoint_path)
            print('loaded the model from checkpoint_path: ' + global_checkpoint_path + '----')
            with open(resume_file) as f:
                res = json.load(f)
                res_itr = res['resitr']
            print('loading training from previous iter at: ',res_itr)
        else:
            print('----check_point path does not exist: ' + global_checkpoint_path + ' , starts training from scratch----')
            timestamp = {}
            timestamp['resitr'] = 1
            with open(resume_file,'w') as f:
                json.dump(timestamp,f)
    else:
        print('----restore_checkpoint arg is False, starts training from sratch----')

    # Instantiate the runner object
    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam,
     extrinsic_factor = extrinsic_factor, intrinsic_factor = intrinsic_factor, need_log_imgs=need_log_imgs, model_type=model_type)
    if eval_env is not None:
        eval_runner = Runner(env = eval_env, model = model, nsteps = nsteps, gamma = gamma, lam= lam,
         extrinsic_factor = extrinsic_factor, intrinsic_factor = intrinsic_factor, need_log_imgs=need_log_imgs, model_type=model_type)

    epinfobuf = deque(maxlen=100)
    if eval_env is not None:
        eval_epinfobuf = deque(maxlen=100)

    if init_fn is not None:
        init_fn()

    # Start total timer
    tfirststart = time.perf_counter()

    nupdates = total_timesteps//nbatch
    #nupdates = total_timesteps
    assert log_interval == save_interval
    for update in range(res_itr, nupdates+1):
        assert nbatch % nminibatches == 0
        # Start timer
        tstart = time.perf_counter()
        frac = 1.0 - (update - 1.0) / nupdates
        # Calculate the learning rate
        lrnow = lr(frac)
        # Calculate the cliprange
        cliprangenow = cliprange(frac)

        if update % log_interval == 0: logger.info('Stepping environment...')

        # Get minibatch
        if model_type == 'SURPRISE':
            obs, obs_after, returns, extr_rwds,intr_rwds, masks, actions, values, neglogpacs, recls, klds, states, epinfos = runner.run() #pylint: disable=E0632
        else:
            obs, obs_after, returns, extr_rwds,intr_rwds, masks, actions, values, neglogpacs, states, epinfos = runner.run() #pylint: disable=E0632

        if eval_env is not None:
            eval_obs, eval_obs_after, eval_after, eval_returns, eval_extr_rwds,eval_intr_rwds, eval_masks, eval_actions, eval_values, eval_neglogpacs, eval_recls, eval_klds, eval_states, eval_epinfos = eval_runner.run() #pylint: disable=E0632

        if update % log_interval == 0: logger.info('Done.')

        epinfobuf.extend(epinfos)
        if eval_env is not None:
            eval_epinfobuf.extend(eval_epinfos)

        # Here what we're going to do is for each minibatch calculate the loss and append it.
        mblossvals = []
        if states is None: # nonrecurrent version
            # Index of each element of batch_size
            # Create the indices array
            inds = np.arange(nbatch)
            for _ in range(noptepochs):
                # Randomize the indexes
                np.random.shuffle(inds)
                # 0 to batch_size with batch_train_size step
                for start in range(0, nbatch, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start:end]
                    slices = (arr[mbinds] for arr in (obs, obs_after, returns, extr_rwds, intr_rwds, masks, actions, values, neglogpacs))
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices))
        else: # recurrent version
            assert nenvs % nminibatches == 0
            envsperbatch = nenvs // nminibatches
            envinds = np.arange(nenvs)
            flatinds = np.arange(nenvs * nsteps).reshape(nenvs, nsteps)
            for _ in range(noptepochs):
                np.random.shuffle(envinds)
                for start in range(0, nenvs, envsperbatch):
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]
                    mbflatinds = flatinds[mbenvinds].ravel()
                    slices = (arr[mbflatinds] for arr in (obs, obs_after, returns,extr_rwds, intr_rwds, masks, actions, values, neglogpacs))
                    mbstates = states[mbenvinds]
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices, mbstates))

        # Feedforward --> get losses --> update
        lossvals = np.mean(mblossvals, axis=0)
        # End timer
        tnow = time.perf_counter()
        # Calculate the fps (frame per second)
        fps = int(nbatch / (tnow - tstart))

        if update_fn is not None:
            update_fn(update)

        if update % log_interval == 0 or update == 1:
            # Calculates if value function is a good predicator of the returns (ev > 1)
            # or if it's just worse than predicting nothing (ev =< 0)
            ev = explained_variance(values, returns)
            logger.logkv("misc/serial_timesteps", update*nsteps)
            logger.logkv("misc/nupdates", update)
            logger.logkv("misc/total_timesteps", update*nbatch)
            logger.logkv("fps", fps)
            logger.logkv("misc/explained_variance", float(ev))
            logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
            if eval_env is not None:
                logger.logkv('eval_eprewmean', safemean([epinfo['r'] for epinfo in eval_epinfobuf]) )
                logger.logkv('eval_eplenmean', safemean([epinfo['l'] for epinfo in eval_epinfobuf]) )
            logger.logkv('misc/time_elapsed', tnow - tfirststart)
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                print("adding this: ", lossval, lossname)
                logger.logkv('loss/' + lossname, lossval)

            logger.dumpkvs()

        if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir():
            #checkdir = osp.join(logger.get_dir(), 'checkpoints')
            #os.makedirs(checkdir, exist_ok=True)
            #savepath = osp.join(checkdir, '%.5i'%update)
            print('Saving to global checkpoint path: ', global_checkpoint_path)
            model.save(global_checkpoint_path)
            timestamp = {}
            timestamp['resitr'] = update
            with open(resume_file,'w') as f:
                json.dump(timestamp,f)


    return model
# Avoid division error when calculate the mean (in our case if epinfo is empty returns np.nan, not return an error)
def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)
