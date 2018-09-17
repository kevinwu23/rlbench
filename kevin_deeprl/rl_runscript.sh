#!/bin/bash
#SBATCH --gres=gpu:1 			#GPU capabilities
#SBATCH -n 4                    # Number of cores
#SBATCH -N 1                    # Ensure that all cores are on one machine
#SBATCH -t 4-12:00              # Runtime in D-HH:MM
#SBATCH -p gpu       			# Partition to submit to
#SBATCH --mem=64000             # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o output/butterflies-levelswitch-reset3.out      # File to which STDOUT will be written
#SBATCH -e errors/butterflies-levelswitch-reset3.err      # File to which STDERR will be written

source activate pt36
# module load jdk/1.8.0_45-fasrc01
# python main.py -game_name gvgai-missilecommand -doubleq 1 -trial_num 1 -level_switch levelswitch -model_weight_path gvgai-missilecommand_episode2948_trial1_levelswitch.pt -pretrain 1
python main.py -game_name gvgai-butterflies -doubleq 1 -trial_num 1 -level_switch levelswitch -train_mode alllevels -level_to_test 10
# python main.py -game_name expt_antagonist -doubleq 1 -trial_num 1 -level_switch other
# python main.py -game_name gvgai-zelda -doubleq 1 -trial_num 12 -pretrain 0 -level_switch other
# python main.py -game_name gvgai-portals -doubleq 1 -trial_num 12 -lr .0001 -pretrain 0 -level_switch other
# python main.py -game_name seaquest -doubleq 1 -trial_num 12 -lr .00001 -pretrain 0 -level_switch other
