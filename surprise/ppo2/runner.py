import numpy as np
from surprise.common.runners import AbstractEnvRunner
import os
import time
import imageio
import pickle, lzma

class Runner(AbstractEnvRunner):
    """
    We use this object to make a mini batch of experiences
    __init__:
    - Initialize the runner

    run():
    - Make a mini batch
    """
    def __init__(self, *, env, model, nsteps, gamma, lam, extrinsic_factor, intrinsic_factor, need_log_imgs, model_type):
        super().__init__(env=env, model=model, nsteps=nsteps)
        # Lambda used in GAE (General Advantage Estimation)
        self.lam = lam
        # Discount rate
        self.gamma = gamma
        self.e_factor = extrinsic_factor
        self.i_factor = intrinsic_factor
        self.need_log_imgs = need_log_imgs
        self.model_type = model_type

    def run(self):
        # Here, we init the lists that will contain the mb of experiences
        mb_obs, mb_obs_after, mb_rewards, mb_extr_rwd, mb_intr_rwd, mb_actions, mb_values, mb_dones, mb_neglogpacs, mb_klds, mb_recls  = [],[],[],[],[],[],[],[],[],[],[]
        mb_states = self.states
        epinfos = []
        # For n in range number of steps
        last_obs = self.obs[:]

        # env variable to track episode number
        try:
            ep_num = int(os.environ["EPISODE_NUM"]) + 1
        except:
            ep_num = 0
        os.environ["EPISODE_NUM"] = str(ep_num)

        if self.need_log_imgs and (self.model_type=='SURPRISE2' or self.model_type=='SURPRISE'):
            # Set up log files
            test_path = str(os.environ['OPENAI_LOGDIR'])+"/image_logs/"
            os.makedirs(test_path, exist_ok=True)
            start_stamp = '_'.join(time.ctime(time.time()).split(' ')[2:])
            fname = test_path+"episode_"+str(ep_num)+"_starttime_"+start_stamp
            score_log = []
            latent_log = []
            #score_log_file = open(fname+"score.txt","w+")
            #latent_log_file = open(fname+"latent.txt","w+")

        for step_num in range(self.nsteps):
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            actions, values, self.states, neglogpacs = self.model.step(self.obs, S=self.states, M=self.dones)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)

            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            last_obs = self.obs[:]
            self.obs[:], rewards, self.dones, infos = self.env.step(actions)
            #print('obs', self.obs)

            # store the next observation as well for curiosity
            mb_obs_after.append(self.obs.copy())

            # Adds curiosity reward
            # TODO: change the hardcoded action dimension 1 here
            # Note that the dimension is not equal to action_space.n for discrete action_space type
            curiosity_bonus = self.model.predict_curiosity_bonus(last_obs[:], self.obs[:], self.model.actions_to_labels(actions))
            rec_loss = None
            kl_div = None
            if (self.model_type=='SURPRISE'):
                curiosity_bonus, zs, encoded_next, reconstructed_next, rec_loss, kl_div = curiosity_bonus
            #log here
                if self.need_log_imgs and step_num%10==0:
                    c_mean = np.mean(curiosity_bonus)
                    outlier_list = [abs(a-c_mean) for a in curiosity_bonus]
                    idx = outlier_list.index(max(outlier_list))
                    log_intr_rwd = curiosity_bonus[idx]
                    log_extr_rwd = rewards[idx]
                    log_z = zs[idx]
                    log_last_img = last_obs[idx]
                    log_next_img = self.obs[idx]
                    log_rec_img = reconstructed_next[idx]
                    log_enc_next_img = encoded_next[idx]
                    # uncomment to actually log
                    imageio.imwrite(fname+"_step_"+str(step_num)+"_last_obs_img.png", log_last_img)
                    imageio.imwrite(fname+"_step_"+str(step_num)+"_next_obs_img.png", log_next_img)
                    with lzma.open(fname+"_step_"+str(step_num)+"_reconstructed_next_img.pkl.lzma",'wb') as f:
                        pickle.dump(log_rec_img, f)
                    with lzma.open(fname+"_step_"+str(step_num)+"_encoded_next_img.pkl.lzma",'wb') as f:
                        pickle.dump(log_enc_next_img, f)
                    score_log.append((log_intr_rwd,log_extr_rwd))
                    latent_log.append(log_z)
                    #score_log_file.write(str(log_intr_rwd)+" "+str(log_extr_rwd)+"\n")
                    #latent_log_file.write(" ".join(map(str,log_z))+"\n")
                    #print("logged "+str(step_num))

            #instead of if-self statements, multiply intrinsic and extrinsic factors, which
            #will be taken in from command line
            combined_rewards = self.e_factor*rewards + self.i_factor*curiosity_bonus

            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(combined_rewards)
            mb_extr_rwd.append(self.e_factor*rewards)
            mb_intr_rwd.append(self.i_factor*curiosity_bonus)
            mb_klds.append(kl_div)
            mb_recls.append(rec_loss)

        if self.need_log_imgs and (self.model_type=='SURPRISE2' or self.model_type=='SURPRISE'):
            #close log files
            with lzma.open(fname+"score.pkl.lzma","wb") as f:
                pickle.dump(score_log, f)
            with lzma.open(fname+"latent.pkl.lzma","wb") as f:
                pickle.dump(latent_log, f)
            #score_log_file.close()
            #latent_log_file.close()
            #batch of steps to batch of rollouts

        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_obs_after = np.asarray(mb_obs_after, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_intr_rwd = np.asarray(mb_intr_rwd, dtype=np.float32)
        mb_extr_rwd = np.asarray(mb_extr_rwd, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        mb_klds = np.asarray(mb_klds, dtype=np.float32)
        mb_recls = np.asarray(mb_recls, dtype=np.float32)
        last_values = self.model.value(self.obs, S=self.states, M=self.dones)


        # discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values
        if (self.model_type=='SURPRISE'):
            return (*map(sf01, (mb_obs, mb_obs_after, mb_returns, mb_extr_rwd, mb_intr_rwd, mb_dones, mb_actions, mb_values, mb_neglogpacs, mb_recls, mb_klds)),
                mb_states, epinfos)
        return (*map(sf01, (mb_obs, mb_obs_after, mb_returns, mb_extr_rwd, mb_intr_rwd, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
            mb_states, epinfos)
# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])
