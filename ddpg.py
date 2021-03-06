import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from model import *


class DDPG(object):
    def __init__(self, gamma, tau, infeat, outfeat, hfeat1, hfeat2, doubleDQN, policy_update_interval, policy_smooth, max_action, batchnorm, device):
        """hyperparameters"""
        self.infeat = infeat
        self.hfeat1 = hfeat1
        self.hfeat2 = hfeat2
        self.outfeat = outfeat
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.doubleDQN = doubleDQN
        self.policy_update_interval = policy_update_interval
        self.policy_smooth = policy_smooth
        self.max_action = max_action
        self.batchnorm = batchnorm
        self.update_cnt = 0

        """actor"""
        self.actor = Actor(infeat, outfeat, hfeat1, hfeat2,
                           batchnorm).to(self.device)
        self.actor_target = Actor(
            infeat, outfeat, hfeat1, hfeat2, batchnorm).to(self.device)

        """critic"""
        self.critic = Critic(infeat, outfeat, hfeat1,
                             hfeat2, batchnorm).to(self.device)
        self.critic_target = Critic(
            infeat, outfeat, hfeat1, hfeat2, batchnorm).to(self.device)
        if self.doubleDQN:
            self._critic = Critic(infeat, outfeat, hfeat1,
                                  hfeat2, batchnorm).to(self.device)
            self._critic_target = Critic(
                infeat, outfeat, hfeat1, hfeat2, batchnorm).to(self.device)

        """optimizer"""
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=1e-3)
        if self.doubleDQN:
            self._critic_optimizer = torch.optim.Adam(
                self._critic.parameters(), lr=1e-3)

        """hard update"""
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        if self.doubleDQN:
            self._critic_target.load_state_dict(self._critic.state_dict())

    def act(self, state, ounoise=None, pattern="train"):
        """????????????"""
        with torch.no_grad():
            self.actor.eval()
            action = self.actor(state.to(self.device))

            """????????????"""
            self.actor.train()
            if pattern == "train":
                action += torch.tensor(ounoise.noise()).to(self.device)
            action = action.clamp(-1, 1)

        return action

    def update(self, batch):
        """??????????????????"""
        """??? batch ???????????????"""
        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)
        mask_batch = torch.cat(batch.mask).to(self.device)
        next_state_batch = torch.cat(batch.next_state).to(self.device)

        """actor ?????????????????? action"""
        next_action_batch = self.actor_target(next_state_batch)
        """????????????, ?????? Target policy smoothing regularization"""
        if self.policy_smooth:
            """????????????????????????, ??????????????????????????????"""
            noise = torch.ones_like(action_batch).data.normal_(
                0, 0.2).to(self.device)
            noise = noise.clamp(-0.5, 0.5)
            next_action_batch += noise
            if self.max_action is not None:
                next_action_batch.clamp(-self.max_action, self.max_action)

        """critic??????action????????????"""
        next_state_action_values = self.critic_target(
            next_state_batch, next_action_batch).detach()
        
        """?????????double DQN, ???????????????Critic????????????, ????????????????????????????????? Q ???"""
        if self.doubleDQN:
            _next_state_action_values = self._critic_target(
                next_state_batch, next_action_batch)
            next_state_action_values = torch.min(
                next_state_action_values, _next_state_action_values).detach()

        """??????reward????????????"""
        reward_batch = reward_batch.unsqueeze(1)
        mask_batch = mask_batch.unsqueeze(1)
        expected_state_action_batch = reward_batch + \
            (self.gamma * mask_batch * next_state_action_values)

        """??????critic????????????"""
        self.critic_optimizer.zero_grad()

        state_action_batch = self.critic((state_batch), (action_batch))

        value_loss = F.mse_loss(
            state_action_batch, expected_state_action_batch)
        value_loss.backward()
        self.critic_optimizer.step()

        """?????????double DQN, ???????????????????????????????????????"""
        if self.doubleDQN:
            self._critic_optimizer.zero_grad()

            _state_action_batch = self._critic((state_batch), (action_batch))

            _value_loss = F.mse_loss(
                _state_action_batch, expected_state_action_batch)
            _value_loss.backward()
            self._critic_optimizer.step()

        """??????critic????????????actor????????????"""
        if self.update_cnt % self.policy_update_interval == 0:
            self.actor_optimizer.zero_grad()

            policy_loss = -self.critic((state_batch),
                                       self.actor((state_batch)))

            policy_loss = policy_loss.mean()
            policy_loss.backward()
            self.actor_optimizer.step()

        """??????, ??????actor???delay??????"""
        self.update_cnt += 1

        """???????????????"""
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        if self.doubleDQN:
            soft_update(self._critic_target, self._critic, self.tau)
            return value_loss.item(), _value_loss.item()

        return value_loss.item()
