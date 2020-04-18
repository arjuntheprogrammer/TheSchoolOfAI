# Importing the libraries
import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import deque

action2rotation = np.array([0,5,10,-5,-10])
it = 0

class ReplayBuffer(object):
  def __init__(self, max_size=50000):
    self.storage = []
    self.max_size = max_size
    self.ptr = 0

  def add(self, transition):
    if len(self.storage) == self.max_size:
      self.ptr = (self.ptr) % self.max_size
      self.storage[int(self.ptr)] = transition
    else:
      self.storage.append(transition)
    self.ptr+=1

  def sample(self, batch_size):
    ind = np.random.randint(0, len(self.storage), size=batch_size)
    batch_states_img, batch_states_ori1, batch_states_ori2, batch_states_dis,\
    batch_next_states_img, batch_next_states_ori1, batch_next_states_ori2, batch_next_states_dis,\
    batch_actions, batch_rewards, batch_dones = [], [], [], [], [], [], [], [], [], [], []

    for i in ind:
      state_img,state_ori1,state_ori2,state_dis,\
      next_state_img,next_state_ori1,next_state_ori2,next_state_dis,\
      action, reward, done = self.storage[i]

      batch_states_img.append(np.array(state_img, copy=False))
      batch_states_ori1.append(np.array(state_ori1, copy=False))
      batch_states_ori2.append(np.array(state_ori2, copy=False))
      batch_states_dis.append(np.array(state_dis, copy=False))

      batch_next_states_img.append(np.array(next_state_img, copy=False))
      batch_next_states_ori1.append(np.array(next_state_ori1, copy=False))
      batch_next_states_ori2.append(np.array(next_state_ori2, copy=False))
      batch_next_states_dis.append(np.array(next_state_dis, copy=False))


      batch_actions.append(action)
      batch_rewards.append(np.array(reward, copy=False))
      batch_dones.append(np.array(done, copy=False))

    return batch_states_img, batch_states_ori1, batch_states_ori2, batch_states_dis,\
    batch_next_states_img, batch_next_states_ori1, batch_next_states_ori2, batch_next_states_dis,\
    batch_actions, np.array(batch_rewards).reshape(-1, 1), np.array(batch_dones).reshape(-1, 1)

class Actor(nn.Module):
  def __init__(self, state_dim=5, action_dim=5, max_action=1):
    super(Actor, self).__init__()
    self.conv1 = nn.Conv2d(1,3,kernel_size = 3)
    self.conv2 = nn.Conv2d(3, 20, kernel_size=3)
    self.conv2_drop = nn.Dropout2d()

    self.fc1 = nn.Linear(2420, 5)
    self.fc2 = nn.Linear(8, 100)
    self.fc3 = nn.Linear(100, action_dim)
    self.max_action = max_action

  def forward(self, x_img, x_ori1, x_ori2, x_dis):
    x = F.relu(F.max_pool2d(self.conv1(x_img), 2))
    x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
    x = x.reshape(x.size(0), -1) # 2420
    x = F.relu(self.fc1(x)) # 5
    x = torch.cat([x, x_ori1, x_ori2, x_dis], 1) #8
    x = F.relu(self.fc2(x)) # 100
    x = F.dropout(x, training=self.training)
    x = self.max_action * torch.tanh(self.fc3(x)) #3

    values, indices = x.max(1)
    indices = indices.numpy()[:]
    rotation = action2rotation[indices]
    return torch.Tensor(np.expand_dims(rotation, axis=1))

# Step 3: We build two neural networks for the two Critic models
# and two neural networks for the two Critic targets
class Critic(nn.Module):

  def __init__(self, state_dim=5, action_dim=5):
    super(Critic, self).__init__()

    self.conv1 = nn.Conv2d(1,3,kernel_size = 3)
    self.conv2 = nn.Conv2d(3, 20, kernel_size=3)
    self.conv2_drop = nn.Dropout2d()
    self.fc1 = nn.Linear(2420, 5)
    #concat
    self.fc2 = nn.Linear(9, 300)
    self.fc3 = nn.Linear(300, 1)

    self.conv3 = nn.Conv2d(1,3,kernel_size = 3)
    self.conv4 = nn.Conv2d(3, 20, kernel_size=3)
    self.conv4_drop = nn.Dropout2d()
    self.fc4 = nn.Linear(2420, 5)
    #concat
    self.fc5 = nn.Linear(9, 300)
    self.fc6 = nn.Linear(300, 1)


  def forward(self, state_img, state_ori1, state_ori2, state_dis, action):
    x1 = F.relu(F.max_pool2d(self.conv1(state_img), 2))
    x1 = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x1)), 2))
    x1 = x1.reshape(x1.size(0), -1) #2420
    x1 = F.relu(self.fc1(x1)) # 5
    x1 = torch.cat([x1, state_ori1, state_ori2, state_dis, action], 1) # 9
    x1 = F.relu(self.fc2(x1)) #300
    x1 = F.dropout(x1, training=self.training)
    x1 = self.fc3(x1) # 1
    #################
    x2 = F.relu(F.max_pool2d(self.conv3(state_img), 2))
    x2 = F.relu(F.max_pool2d(self.conv4_drop(self.conv4(x2)), 2))
    x2 = x2.reshape(x2.size(0), -1) # 2420
    x2 = F.relu(self.fc4(x2)) # 5
    x2 = torch.cat([x2, state_ori1, state_ori2, state_dis, action], 1) # 9
    x2 = F.relu(self.fc5(x2)) #300
    x2 = F.dropout(x2, training=self.training)
    x2 = self.fc6(x2) # 1

    return x1, x2

  def Q1(self, state_img, state_ori1, state_ori2, state_dis, action):
    x1 = F.relu(F.max_pool2d(self.conv1(state_img), 2))
    x1 = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x1)), 2))
    x1 = x1.reshape(x1.size(0), -1) #2420
    x1 = F.relu(self.fc1(x1)) # 5
    x1 = torch.cat([x1, state_ori1, state_ori2, state_dis, action], 1) # 9
    x1 = F.relu(self.fc2(x1)) #300
    x1 = F.dropout(x1, training=self.training)
    x1 = self.fc3(x1) # 3
    return x1

# Building the whole Training Process into a class
class TD3(object):

  def __init__(self, state_dim, action_dim, max_action):
    self.actor = Actor(state_dim, action_dim, max_action)
    self.actor_target = Actor(state_dim, action_dim, max_action)
    self.actor_target.load_state_dict(self.actor.state_dict())
    self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
    self.critic = Critic(state_dim, action_dim)
    self.critic_target = Critic(state_dim, action_dim)
    self.critic_target.load_state_dict(self.critic.state_dict())
    self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
    self.max_action = max_action

  def select_action(self, state_img,state_ori1,state_ori2,state_dis):
    state_img = torch.Tensor(np.expand_dims(state_img, axis=0))
    state_img = torch.Tensor(np.expand_dims(state_img, axis=0))
    state_ori1 = torch.Tensor(np.expand_dims(state_ori1, axis=0))
    state_ori1 = torch.Tensor(np.expand_dims(state_ori1, axis=0))
    state_ori2 = torch.Tensor(np.expand_dims(state_ori2, axis=0))
    state_ori2 = torch.Tensor(np.expand_dims(state_ori2, axis=0))
    state_dis = torch.Tensor(np.expand_dims(state_dis, axis=0))
    state_dis = torch.Tensor(np.expand_dims(state_dis, axis=0))
    return self.actor(state_img,state_ori1,state_ori2,state_dis).cpu().data.numpy().flatten()

  def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
    global it
    it += 1

    # Step 4: We sample a batch of transitions (s, s', a, r) from the memory
    batch_states_img,batch_states_ori1,batch_states_ori2,batch_states_dis,\
    batch_next_states_img,batch_next_states_ori1,batch_next_states_ori2,batch_next_states_dis,\
    batch_actions, batch_rewards, batch_dones = replay_buffer.sample(batch_size)

    state_img = torch.Tensor(np.expand_dims(batch_states_img, axis=1))
    state_ori1 = torch.Tensor(np.expand_dims(batch_states_ori1, axis=1))
    state_ori2 = torch.Tensor(np.expand_dims(batch_states_ori2, axis=1))
    state_dis = torch.Tensor(np.expand_dims(batch_states_dis, axis=1))

    next_state_img = torch.Tensor(np.expand_dims(batch_next_states_img, axis=1))
    next_state_ori1 = torch.Tensor(np.expand_dims(batch_next_states_ori1, axis=1))
    next_state_ori2 = torch.Tensor(np.expand_dims(batch_next_states_ori2, axis=1))
    next_state_dis = torch.Tensor(np.expand_dims(batch_next_states_dis, axis=1))

    action = torch.Tensor(np.expand_dims(batch_actions, axis=1))
    reward = torch.Tensor(batch_rewards)
    done = torch.Tensor(batch_dones)

    # Step 5: From the next state s', the Actor target plays the next action a'
    next_action = self.actor_target(next_state_img, next_state_ori1, next_state_ori2, next_state_dis)

    # # Step 6: We add Gaussian noise to this next action a' and we clamp it in a range of values supported by the environment
    # noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise)
    # noise = noise.clamp(-noise_clip, noise_clip)
    # next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

    # Step 7: The two Critic targets take each the couple (s', a') as input and return two Q-values Qt1(s',a') and Qt2(s',a') as outputs
    target_Q1, target_Q2 = self.critic_target(next_state_img, next_state_ori1, next_state_ori2, next_state_dis, next_action)

    # Step 8: We keep the minimum of these two Q-values: min(Qt1, Qt2)
    target_Q = torch.min(target_Q1, target_Q2)

    # Step 9: We get the final target of the two Critic models
    target_Q = reward + ((1 - done) * discount * target_Q).detach()

    # Step 10: The two Critic models take each the couple (s, a) as input and return two Q-values Q1(s,a) and Q2(s,a) as outputs
    current_Q1, current_Q2 = self.critic(state_img,state_ori1,state_ori2,state_dis, action)

    # Step 11: We compute the loss coming from the two Critic models: Critic Loss = MSE_Loss(Q1(s,a), Qt) + MSE_Loss(Q2(s,a), Qt)
    critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

    # Step 12: We backpropagate this Critic loss and update the parameters of the two Critic models with a SGD optimizer
    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    self.critic_optimizer.step()

    # Step 13: Once every two iterations, we update our Actor model by performing gradient ascent on the output of the first Critic model
    if it % policy_freq == 0:
      actor_loss = -self.critic.Q1(state_img,state_ori1,state_ori2,state_dis, self.actor(state_img,state_ori1,state_ori2,state_dis)).mean()
      self.actor_optimizer.zero_grad()
      actor_loss.backward()
      self.actor_optimizer.step()

      # Step 14: Still once every two iterations, we update the weights of the Actor target by polyak averaging
      for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

      # Step 15: Still once every two iterations, we update the weights of the Critic target by polyak averaging
      for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
        target_param.data.copy_(tau * param.data + (1 -  tau) * target_param.data)

  # Making a save method to save a trained model
  def save(self, filename, directory):
    torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
    torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))

  # Making a load method to load a pre-trained model
  def load(self, filename, directory):
    self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
    self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))