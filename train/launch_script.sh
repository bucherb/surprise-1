#!/bin/bash

seed_num=$1 # seed number for multiple runs with same config
int_weight=$2 # intrinsic weight e.g. 0.1
ext_weight=$3 # extrinsic weight e.g. 0.9
kl_div_w=$4
rec_loss_w=$5
inv_loss_w=$6
model=$7

# The 8 Atari games are split across four pods
base_name="${model/./}-seed${seed_num/./}-if${int_weight/./}-ef${ext_weight/./}-kl${kl_div_w/./}-rec${rec_loss_w/./}-inv${inv_loss_w/./}"

name="${base_name/./}-1"
echo save-$name
kcreator \
  -g 1 \
  --job-name $name \
  -w /NAS/home/surprise \
  -nc 1 \
  -b 48 \
  -t 3600 \
  -- /bin/bash /NAS/home/surprise/train/train_int_ext_1.sh $seed_num $int_weight $ext_weight $kl_div_w $rec_loss_w $inv_loss_w $model\

krun $name.yaml

name2="${base_name/./}-2"
echo save-$name2
kcreator \
  -g 1 \
  --job-name $name2 \
  -w /NAS/home/surprise \
  -nc 1 \
  -b 48 \
  -t 3600 \
  -- /bin/bash /NAS/home/surprise/train/train_int_ext_2.sh $seed_num $int_weight $ext_weight $kl_div_w $rec_loss_w $inv_loss_w $model\

krun $name2.yaml

name3="${base_name/./}-3"
echo save-$name3
kcreator \
  -g 1 \
  --job-name $name3 \
  -w /NAS/home/surprise \
  -nc 1 \
  -b 48 \
  -t 3600 \
  -- /bin/bash /NAS/home/surprise/train/train_int_ext_3.sh $seed_num $int_weight $ext_weight $kl_div_w $rec_loss_w $inv_loss_w $model\

krun $name3.yaml

name4="${base_name/./}-4"
echo save-$name4
kcreator \
  -g 1 \
  --job-name $name4 \
  -w /NAS/home/surprise \
  -nc 1 \
  -b 48 \
  -t 3600 \
  -- /bin/bash /NAS/home/surprise/train/train_int_ext_4.sh $seed_num $int_weight $ext_weight $kl_div_w $rec_loss_w $inv_loss_w $model\

krun $name4.yaml

kubectl get pods
