#!/bin/bash

#SBATCH --partition=long                                 # Ask for unkillable job
#SBATCH --cpus-per-task=4                                # Ask for 4 CPUs
#SBATCH --gres=gpu:1                                     # Ask for 1 GPU
#SBATCH --mem=40G                                        # Ask for 10 GB of RAM
#SBATCH --time=72:00:00                                   # The job will run for 3 hours
#SBATCH -o /network/scratch/o/omar.salemohamed/wandb-sweep-%j.out  # Write the log on scratch

# 1. Load the required modules
# module --quiet load python/3.9
module unload python
module load anaconda/3

# 2. Load environment
# source ~/.virtualenvs/tasksim/bin/activate
conda activate urlb

# # 3. Copy your dataset on the compute node
# cp /network/data/<dataset> $SLURM_TMPDIR

# 4. Launch your job, tell it to save the model in $SLURM_TMPDIR
#    and look for the dataset into $SLURM_TMPDIR

# python3.8 pretrain.py agent=rnd domain=walker obs_type=pixels
# python3.8 pretrain.py agent=rnd domain=walker obs_type=pixels action_repeat=2 batch_size=256  num_train_frames=100010
# python3.8 pretrain.py agent=hyper_rnd domain=walker obs_type=pixels action_repeat=2 batch_size=256  num_train_frames=100010
# python3.8 pretrain.py agent=hyper_rnd domain=walker obs_type=pixels action_repeat=2 batch_size=256  num_train_frames=10010
# python3.8 pretrain.py agent=hyper_icm_apt domain=walker obs_type=pixels action_repeat=2 batch_size=256  num_train_frames=100010 seed=1



# python3.8 finetune.py agent=rnd task=walker_walk snapshot_ts=100000 obs_type=pixels action_repeat=2 batch_size=256  num_train_frames=100010
# python3.8 finetune.py agent=hyper_rnd task=walker_walk snapshot_ts=100000 obs_type=pixels action_repeat=2 batch_size=256  num_train_frames=100010
# python3.8 finetune.py agent=hyper_ddpg task=walker_walk snapshot_ts=0 obs_type=pixels action_repeat=2 batch_size=256  num_train_frames=300010

## EXP 1
# python3.8 pretrain.py agent=rnd domain=walker obs_type=pixels action_repeat=2 batch_size=256  num_train_frames=100010 seed=1
python3.8 finetune.py agent=rnd task=walker_stand snapshot_ts=100000 obs_type=pixels action_repeat=2 batch_size=256  num_train_frames=100010 seed=1
# python3.8 finetune.py agent=rnd task=walker_walk snapshot_ts=100000 obs_type=pixels action_repeat=2 batch_size=256  num_train_frames=100010 seed=1
# python3.8 finetune.py agent=rnd task=walker_run snapshot_ts=100000 obs_type=pixels action_repeat=2 batch_size=256  num_train_frames=100010 seed=1
# python3.8 finetune.py agent=rnd task=walker_flip snapshot_ts=100000 obs_type=pixels action_repeat=2 batch_size=256  num_train_frames=100010 seed=1

## EXP 2
# python3.8 pretrain.py agent=hyper_rnd domain=walker obs_type=pixels action_repeat=2 batch_size=256  num_train_frames=100010 seed=1
# python3.8 finetune.py agent=hyper_rnd task=walker_stand snapshot_ts=100000 obs_type=pixels action_repeat=2 batch_size=256  num_train_frames=100010 seed=1
# python3.8 finetune.py agent=hyper_rnd task=walker_walk snapshot_ts=100000 obs_type=pixels action_repeat=2 batch_size=256  num_train_frames=100010 seed=1
# python3.8 finetune.py agent=hyper_rnd task=walker_run snapshot_ts=100000 obs_type=pixels action_repeat=2 batch_size=256  num_train_frames=100010 seed=1
# python3.8 finetune.py agent=hyper_rnd task=walker_flip snapshot_ts=100000 obs_type=pixels action_repeat=2 batch_size=256  num_train_frames=100010 seed=1

## EXP 3
# python3.8 pretrain.py agent=icm_apt domain=walker obs_type=pixels action_repeat=2 batch_size=256  num_train_frames=100010 seed=1
# python3.8 finetune.py agent=icm_apt task=walker_stand snapshot_ts=100000 obs_type=pixels action_repeat=2 batch_size=256  num_train_frames=100010 seed=1
# python3.8 finetune.py agent=icm_apt task=walker_walk snapshot_ts=100000 obs_type=pixels action_repeat=2 batch_size=256  num_train_frames=100010 seed=1
# python3.8 finetune.py agent=icm_apt task=walker_run snapshot_ts=100000 obs_type=pixels action_repeat=2 batch_size=256  num_train_frames=100010 seed=1
# python3.8 finetune.py agent=icm_apt task=walker_flip snapshot_ts=100000 obs_type=pixels action_repeat=2 batch_size=256  num_train_frames=100010 seed=1

# ## EXP 4
# python3.8 pretrain.py agent=hyper_icm_apt domain=walker obs_type=pixels action_repeat=2 batch_size=256  num_train_frames=100010 seed=1
# python3.8 finetune.py agent=hyper_icm_apt task=walker_stand snapshot_ts=100000 obs_type=pixels action_repeat=2 batch_size=256  num_train_frames=100010 seed=1
# python3.8 finetune.py agent=hyper_icm_apt task=walker_walk snapshot_ts=100000 obs_type=pixels action_repeat=2 batch_size=256  num_train_frames=100010 seed=1
# python3.8 finetune.py agent=hyper_icm_apt task=walker_run snapshot_ts=100000 obs_type=pixels action_repeat=2 batch_size=256  num_train_frames=100010 seed=1
# python3.8 finetune.py agent=hyper_icm_apt task=walker_flip snapshot_ts=100000 obs_type=pixels action_repeat=2 batch_size=256  num_train_frames=100010 seed=1

## EXP 5
python3.8 pretrain.py agent=hyper_ddpg_rnd domain=walker obs_type=pixels action_repeat=2 batch_size=256  num_train_frames=100010 seed=1
python3.8 finetune.py agent=hyper_ddpg_rnd task=walker_stand snapshot_ts=100000 obs_type=pixels action_repeat=2 batch_size=256  num_train_frames=100010 seed=1
python3.8 finetune.py agent=hyper_ddpg_rnd task=walker_walk snapshot_ts=100000 obs_type=pixels action_repeat=2 batch_size=256  num_train_frames=100010 seed=1
python3.8 finetune.py agent=hyper_ddpg_rnd task=walker_run snapshot_ts=100000 obs_type=pixels action_repeat=2 batch_size=256  num_train_frames=100010 seed=1
python3.8 finetune.py agent=hyper_ddpg_rnd task=walker_flip snapshot_ts=100000 obs_type=pixels action_repeat=2 batch_size=256  num_train_frames=100010 seed=1

# # 5. Copy whatever you want to save on $SCRATCH
# cp $SLURM_TMPDIR/<to_save> /network/scratch/<u>/<username>/