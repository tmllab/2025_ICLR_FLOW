"""PPO (Proximal Policy Optimization) Trainer for MARL-Flow"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass


@dataclass
class PPOConfig:
    """PPO training configuration."""
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    batch_size: int = 64
    normalize_advantages: bool = True


class PPOTrainer:
    """PPO Trainer for individual agents.

    Implements the PPO-Clip algorithm with support for:
    - Generalized Advantage Estimation (GAE)
    - Value function clipping
    - Entropy regularization
    """

    def __init__(self, config: PPOConfig = None):
        self.config = config or PPOConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def compute_returns_and_advantages(self, trajectories: List[Dict],
                                       value_net: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute returns and GAE advantages.

        Args:
            trajectories: List of trajectory data with states, rewards, dones
            value_net: Value network for computing value estimates

        Returns:
            returns: Discounted returns tensor
            advantages: GAE advantages tensor
        """
        states = torch.stack([t['state'] for t in trajectories]).to(self.device)
        rewards = torch.tensor([t['reward'] for t in trajectories],
                              dtype=torch.float32, device=self.device)
        dones = torch.tensor([t.get('done', False) for t in trajectories],
                            dtype=torch.float32, device=self.device)

        # Get value estimates
        with torch.no_grad():
            values = value_net(states).squeeze(-1)

        # Compute GAE
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        gae = 0.0
        next_value = 0.0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = 0.0
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]

            delta = rewards[t] + self.config.gamma * next_value * next_non_terminal - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * next_non_terminal * gae
            advantages[t] = gae
            returns[t] = gae + values[t]

        # Normalize advantages
        if self.config.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return returns, advantages

    def train(self, agent, trajectories: List[Dict]) -> Dict[str, float]:
        """Train agent using PPO.

        Args:
            agent: Agent with policy_net, value_net, and optimizer
            trajectories: Collected trajectory data

        Returns:
            Training metrics dictionary
        """
        if not trajectories:
            return {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0}

        agent.train_mode()

        # Prepare data
        states = torch.stack([t['state'].squeeze(0) if t['state'].dim() > 1 else t['state']
                             for t in trajectories]).to(self.device)
        actions = self._prepare_actions(trajectories)
        old_log_probs = torch.stack([t['log_prob'] for t in trajectories]).to(self.device)

        # Compute returns and advantages
        returns, advantages = self.compute_returns_and_advantages(trajectories, agent.value_net)

        # Training metrics
        policy_losses = []
        value_losses = []
        entropies = []
        kl_divs = []

        # PPO epochs
        for epoch in range(self.config.ppo_epochs):
            # Mini-batch training
            indices = np.random.permutation(len(trajectories))

            for start in range(0, len(trajectories), self.config.batch_size):
                end = min(start + self.config.batch_size, len(trajectories))
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = {k: v[batch_indices] for k, v in actions.items()}
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]

                # Forward pass
                policy_output = agent.policy_net(batch_states)
                values = agent.value_net(batch_states).squeeze(-1)

                # Compute new log probs and entropy
                new_log_probs, entropy = self._evaluate_actions(
                    policy_output, batch_actions
                )

                # Policy loss (PPO-Clip)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(
                    ratio,
                    1 - self.config.clip_ratio,
                    1 + self.config.clip_ratio
                ) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values, batch_returns)

                # Entropy loss
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss +
                    self.config.value_coef * value_loss +
                    self.config.entropy_coef * entropy_loss
                )

                # Backward pass
                agent.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                nn.utils.clip_grad_norm_(
                    agent.get_parameters(),
                    self.config.max_grad_norm
                )

                agent.optimizer.step()

                # Record metrics
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.mean().item())

                # Approximate KL divergence
                with torch.no_grad():
                    kl = (batch_old_log_probs - new_log_probs).mean()
                    kl_divs.append(kl.item())

        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy': np.mean(entropies),
            'kl_divergence': np.mean(kl_divs)
        }

    def _prepare_actions(self, trajectories: List[Dict]) -> Dict[str, torch.Tensor]:
        """Prepare action tensors from trajectories."""
        actions = {}

        # Check what actions are available
        sample = trajectories[0]['action']

        if isinstance(sample, dict):
            for key in sample.keys():
                if key.endswith('_idx') or key.endswith('_action'):
                    actions[key] = torch.tensor(
                        [t['action'][key] for t in trajectories],
                        device=self.device
                    )
        else:
            actions['action'] = torch.tensor(
                [t['action'] for t in trajectories],
                device=self.device
            )

        return actions

    def _evaluate_actions(self, policy_output: Dict[str, torch.Tensor],
                         actions: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate actions to get log probs and entropy."""
        from torch.distributions import Categorical

        log_probs = []
        entropies = []

        # Handle different policy heads
        if 'workflow_probs' in policy_output and 'workflow_idx' in actions:
            dist = Categorical(policy_output['workflow_probs'])
            log_probs.append(dist.log_prob(actions['workflow_idx']))
            entropies.append(dist.entropy())

        if 'schedule_probs' in policy_output and 'schedule_priority' in actions:
            dist = Categorical(policy_output['schedule_probs'])
            log_probs.append(dist.log_prob(actions['schedule_priority']))
            entropies.append(dist.entropy())

        if 'strategy_probs' in policy_output and 'strategy_idx' in actions:
            dist = Categorical(policy_output['strategy_probs'])
            log_probs.append(dist.log_prob(actions['strategy_idx']))
            entropies.append(dist.entropy())

        if 'retry_probs' in policy_output and 'retry_action' in actions:
            dist = Categorical(policy_output['retry_probs'])
            log_probs.append(dist.log_prob(actions['retry_action']))
            entropies.append(dist.entropy())

        # Combine log probs and entropies
        if log_probs:
            total_log_prob = torch.stack(log_probs, dim=-1).sum(dim=-1)
            total_entropy = torch.stack(entropies, dim=-1).mean(dim=-1)
        else:
            # Fallback for simple action space
            if 'action' in actions:
                for key in policy_output:
                    if 'probs' in key:
                        dist = Categorical(policy_output[key])
                        total_log_prob = dist.log_prob(actions['action'])
                        total_entropy = dist.entropy()
                        break

        return total_log_prob, total_entropy


class RolloutBuffer:
    """Buffer for storing trajectory data."""

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(self, state: torch.Tensor, action: Any, reward: float,
            log_prob: torch.Tensor, done: bool = False):
        """Add a transition to the buffer."""
        transition = {
            'state': state,
            'action': action,
            'reward': reward,
            'log_prob': log_prob,
            'done': done
        }

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition

        self.position = (self.position + 1) % self.capacity

    def get_all(self) -> List[Dict]:
        """Get all transitions."""
        return self.buffer

    def clear(self):
        """Clear the buffer."""
        self.buffer = []
        self.position = 0

    def __len__(self):
        return len(self.buffer)
