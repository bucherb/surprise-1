#!/bin/bash

pip install --user imageio
pip install --user gym[atari]
export OPENAI_LOG_FORMAT='tensorboard'
python3 -m surprise.run --alg=ppo2 --env="PitfallNoFrameskip-v4" --num_timesteps=1e8 --model_type=ICM --intrinsic_factor=0.5 --extrinsic_factor=0.5 --global_checkpoint_path="./global_checkpoint/check_log2/ckp000000" --need_log_imgs=False --log_interval=10 --save_interval=10 --logdir="./check_log2/" --resume_file="tmp.txt"
