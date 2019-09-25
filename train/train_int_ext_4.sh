#!/bin/bash

pip install --user imageio
pip install --user gym[atari]

declare -a GameList=("SpaceInvaders"
	"BeamRider")

export OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard'

run_game (){
	local game_name="$1"
	local seed_num="seed$2"
	local intrinsic_factor="$3"
	local extrinsic_factor="$4"
	local kl_div_w="$5"
	local rec_loss_w="$6"
	local inv_loss_w="$7"
	local Model="$8"
	local save_name="${Model}-${game_name}-int${intrinsic_factor/./}-ext${extrinsic_factor/./}-kl${kl_div_w/./}-rec${rec_loss_w/./}-inv${inv_loss_w/./}-${seed_num}"

	echo $save_name

	python3 -m surprise.run --alg=ppo2 --env="${game_name}NoFrameskip-v4" --num_timesteps=1e8 --model_type="${Model}" \
		--intrinsic_factor=$intrinsic_factor --extrinsic_factor=$extrinsic_factor \
		--global_checkpoint_path="/NAS/home/models/global_checkpoint/${save_name}/ckp000000" \
		--need_log_imgs=False --log_interval=10 --save_interval=10 --logdir="/NAS/home/logs/${save_name}/" \
		--resume_file="/NAS/home/models/${save_name}.txt" --kl_div_w=${kl_div_w} --rec_loss_w=${rec_loss_w} --inv_loss_w=${inv_loss_w} --stick_prob=0.25 --nfskip=2 --nfstick=2
}

for game in ${GameList[@]}
do
	run_game $game $1 $2 $3 $4 $5 $6 $7 &
done

sleep 359990
