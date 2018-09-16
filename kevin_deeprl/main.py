
import argparse
import gym
import gym_gvgai
from player import Player
import pdb
import numpy as np

parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
parser.add_argument('-trial_num', default = 1, required = False)
parser.add_argument('-batch_size', default = 32, required = False)
parser.add_argument('-lr', default = .00025, type = float, required = False)
parser.add_argument('-gamma', default = .99, required = False)
parser.add_argument('-eps_start', default = 1, required = False)
parser.add_argument('-eps_end', default = .1, required = False)
parser.add_argument('-eps_decay', default = 200., required = False)
parser.add_argument('-target_update', default = 10, required = False)
parser.add_argument('-img_size', default = 64, required = False)
parser.add_argument('-num_episodes', default = 20000, type = int, required = False)
parser.add_argument('-max_steps', default = 5e6, required = False)
parser.add_argument('-max_mem', default = 50000, required = False)
parser.add_argument('-model_name', default = 'DQN', required = False)
parser.add_argument('-model_weight_path', default = 'gvgai-aliens_episode157_trial1_levelswitch.pt', required = False)
parser.add_argument('-test_mode', default = 0, type = int, required = False)
parser.add_argument('-pretrain', default = 0, type = int, required = False)
parser.add_argument('-cuda', default = 1, required = False)
parser.add_argument('-doubleq', default = 1, type = int, required = False)
parser.add_argument('-level_switch', default = 'random', type = str, required = False)
parser.add_argument('-steps_to_restart', default = 1000, type = int, required = False)
parser.add_argument('-game_name', default = 'aliens', required = False)
parser.add_argument('-level_to_test', default = 3, type = int, required = False)
parser.add_argument('-train_mode', default = 'all_levels', type = str, required = True)

# pdb.set_trace()
# python main.py -game_name gvgai-aliens-lvl0-v0

config = parser.parse_args();

all_gvgai_games = [env.id for env in gym.envs.registry.all() if env.id.startswith('gvgai')]

config.all_level_names = [game for game in all_gvgai_games if config.game_name in game]
config.level_names = config.all_level_names
# config.level_names = [game for game in config.all_level_names if 'lvl0' in game or 'lvl1' in game or 'lvl2' in game]
# config.level_names = [game for game in config.all_level_names if 'lvl3' in game or 'lvl4' in game]


# for game_name in all_gvgai_games:
# 	print(game_name)
# 	print("_"*10)
# pdb.set_trace()

# print(config.game_name)

# gvgai-missilecommand-lvl0-v0

# module load jdk/1.8.0_45-fasrc01

print("Trial {}".format(config.trial_num))
print("Game: {}".format(config.game_name))




game_player = Player(config)
# game_player.env.reset()
# pdb.set_trace()

game_player.train_model()
# game_player.test_model()























