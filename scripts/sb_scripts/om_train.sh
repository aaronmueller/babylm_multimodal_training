#!/bin/bash 
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100G
#SBATCH -t 2-0:0:0
#SBATCH -c 10
#SBATCH -o /om2/user/chengxuz/sbatch_logs/slurm_%j.out

. ~/.bash_conda_init_om2
. ~/.babylm_init
conda activate babylm_clean
. ./sb_scripts/get_rand_port
python -m torch.distributed.launch --nproc_per_node=1 --master_port=$RANDOM_PORT general_train.py --setting ${SETTING}
