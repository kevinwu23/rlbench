import gym
import gym_gvgai
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import rl_models
import random
from collections import namedtuple
import numpy as np
from PIL import Image
import pdb
from scipy import misc
import imageio
import sys

import matplotlib as plt

class Player(object):

	def __init__(self, config):

		self.config = config

		self.all_envs = [gym.make(level_name).unwrapped for level_name in config.level_names]
		self.current_level = 0
		self.current_env = self.all_envs[self.current_level]
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.game_size = np.shape(self.current_env.render(mode='rgb_array'))
		self.input_channels = self.game_size[2]
		self.n_actions = len(self.current_env.actions)

		self.policy_net = rl_models.rl_model(self)
		self.target_net = rl_models.rl_model(self)
		self.target_net.load_state_dict(self.policy_net.state_dict())

		self.optimizer = optim.Adam(self.policy_net.parameters(), lr = config.lr)
		self.memory = rl_models.ReplayMemory(self.config.max_mem)

		self.Transition = namedtuple('Transition',
						('state', 'action', 'next_state', 'reward'))

		self.steps_done = 0
		self.done = 0

		self.resize = T.Compose([T.ToPILImage(),
						T.Pad((np.max(self.game_size[0:2]) - self.game_size[1], np.max(self.game_size[0:2]) - self.game_size[0])),
						T.Resize((self.config.img_size, self.config.img_size), interpolation=Image.CUBIC),
						T.ToTensor()])

		self.episode_durations = []

		self.num_episodes = config.num_episodes

		self.screen_history = []

	def get_screen(self):
		screen = self.env.render(mode='rgb_array').transpose((2, 0, 1))  
		screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
		screen = torch.from_numpy(screen)
		# Resize, and add a batch dimension (BCHW)
		return self.resize(screen).unsqueeze(0).to(self.device)

	def save_gif(self):
		imageio.mimsave('screens/{}_frame{}_trial{}.gif'.format(self.config.game_name, self.total_steps, self.config.trial_num), self.screen_history)

	def append_gif(self):
		frame = self.env.render(mode='rgb_array')
		self.screen_history.append(frame)

	def save_model(self):
		torch.save(self.target_net.state_dict(),'model_weights/{}_episode{}_trial{}.pt'.format(self.config.game_name, self.episode, self.config.trial_num))

	def load_model(self):
		self.target_net.load_state_dict(torch.load('model_weights/{}'.format(self.config.model_weight_path)))
		self.target_net.load_state_dict(torch.load('model_weights/{}'.format(self.config.model_weight_path)))

	def select_action(self):
		sample = np.random.uniform()
		eps_threshold = self.config.eps_end + (self.config.eps_start - self.config.eps_end) * \
			np.exp(-1. * self.steps_done / self.config.eps_decay)
		self.steps_done += 1.
		if sample > eps_threshold:
			with torch.no_grad():
				return self.policy_net(self.state).max(1)[1].view(1, 1)
		else:
			return torch.tensor([[np.random.choice([0,1])]], device=self.device, dtype=torch.long)

	def optimize_model(self):
		if len(self.memory) < self.config.batch_size:
			return
		transitions = self.memory.sample(self.config.batch_size)
		# Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
		# detailed explanation).
		batch = self.Transition(*zip(*transitions))

		# Compute a mask of non-final states and concatenate the batch elements
		non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
											  batch.next_state)), device=self.device, dtype=torch.uint8)
		non_final_next_states = torch.cat([s for s in batch.next_state
													if s is not None])
		state_batch = torch.cat(batch.state)
		action_batch = torch.cat(batch.action)
		reward_batch = torch.cat(batch.reward)

		# Compute Q(s_t, a) - the model computes Q(s_t), then we select the
		# columns of actions taken
		state_action_values = self.policy_net(state_batch).gather(1, action_batch)

		# Compute V(s_{t+1}) for all next states.
		next_state_values = torch.zeros(self.config.batch_size, device=self.device)

		if self.config.doubleq:
			_, next_state_actions = self.policy_net(non_final_next_states).max(1, keepdim=True)
			# pdb.set_trace()
			next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, next_state_actions).squeeze(1)
			next_state_values = next_state_values.data
			# print("Double Q")
		else:
			next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
			# print("Single Q")
		# Compute the expected Q values

		# next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
		expected_state_action_values = (next_state_values * self.config.gamma) + reward_batch
		# pdb.set_trace()

		# Compute Huber loss
		loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
		# pdb.set_trace()
		self.loss_history = loss
		# print(loss)

		# Optimize the model
		self.optimizer.zero_grad()
		loss.backward()
		for param in self.policy_net.parameters():
			param.grad.data.clamp_(-1, 1)
		self.optimizer.step()


	def train_model(self):

		print("Training Starting")
		print("-"*25)

		if self.config.pretrain:

			pring("Loading Model")

			self.load_model()

		self.steps = 0
		self.episode = 0
		self.best_reward = 0
		self.total_reward = 0
		self.total_reward_history = []

		self.env.reset()
		last_screen = self.get_screen()
		current_screen = self.get_screen()
		self.state = current_screen - last_screen

		self.total_reward_history = []

		while self.steps < self.config.max_steps or self.episode < self.num_episodes:

			self.steps += 1


			# Select and perform an action
			self.action = self.select_action()
			# print(self.action)
			_, self.reward, self.done, self.info = self.env.step(self.action.item())

			self.total_reward += self.reward
			self.total_reward_history.append(self.reward)

			self.reward = max(-1.0, min(self.reward, 1.0))

			self.reward = torch.tensor([self.reward], device = self.device)
			
			# print(self.reward)

			# Observe new state
			last_screen = current_screen
			current_screen = self.get_screen()
			if not self.done:
				self.next_state = current_screen - last_screen
			else:
				self.next_state = None

			# Store the transition in memory
			self.memory.push(self)

			# Move to the next state
			self.state = self.next_state

			# Perform one step of the optimization (on the target network)
			self.optimize_model()

			if self.done:

				self.episode += 1

				print("Average Reward at step {}: {}".format(self.steps, self.total_reward))
				sys.stdout.flush()
				
				# Update the target network
				if self.steps > 1000 and not self.steps % 500:

					self.target_net.load_state_dict(self.policy_net.state_dict())

					if self.total_reward > self.best_reward:
						self.best_reward = self.total_reward
						print("New Best Reward: {}".format(self.best_reward))
						self.save_model()

				self.total_reward = 0
				self.env.reset()
				last_screen = self.get_screen()
				current_screen = self.get_screen()
				self.state = current_screen - last_screen

				plt.plot(self.total_reward_history)
				plt.savefig('reward_history.png')


		pdb.set_trace()



	def test_model(self):

		print("Testing Starting")
		print("-"*25)

		if self.config.pretrain:

			print("Loading Pre-Training")

			self.load_model()

		self.episode = 0
		self.total_reward = 0

		self.env.reset()
		last_screen = self.get_screen()
		current_screen = self.get_screen()
		self.state = current_screen - last_screen

		while self.episode < self.num_episodes:

			# print(self.episode)

			# Select and perform an action
			self.action = self.select_action()
			# print(self.action)
			_, self.reward, self.done, self.info = self.env.step(self.action.item())

			self.total_reward += self.reward.numpy()

			self.reward = torch.tensor([self.reward], device = self.device)
			
			# print(self.reward)

			# Observe new state
			last_screen = current_screen
			current_screen = self.get_screen()
			if not self.done:
				self.next_state = current_screen - last_screen
			else:
				self.next_state = None

			# Store the transition in memory
			self.memory.push(self)

			# Move to the next state
			self.state = self.next_state

			# Perform one step of the optimization (on the target network)

			if self.done:

				self.episode += 1
				print(self.episode)
				print(self.total_reward*1. / self.episode)

				self.env.reset()
				last_screen = self.get_screen()
				current_screen = self.get_screen()
				self.state = current_screen - last_screen

		print("Average Reward: {}".format(self.total_reward*1./self.episode))
		sys.stdout.flush()



