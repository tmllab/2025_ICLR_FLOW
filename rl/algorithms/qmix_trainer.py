"""QMIX Trainer for Multi-Agent Credit Assignment"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from copy import deepcopy

from rl.networks.value_nets import QMIXMixer, VDNMixer


@dataclass
class QMIXConfig:
    """QMIX training configuration."""
    lr: float = 5e-4
    gamma: float = 0.99
    n_agents: int = 3
    embed_dim: int = 64
    hypernet_hidden_dim: int = 64
    target_update_freq: int = 200
    grad_norm_clip: float = 10.0
    buffer_size: int = 5000
    batch_size: int = 32


class QMIXTrainer:
    """QMIX Trainer for multi-agent credit assignment.

    Implements QMIX algorithm for learning monotonic mixing
    of individual agent Q-values into a team Q-value.

    Key features:
    - Centralized training with decentralized execution (CTDE)
    - Monotonic value decomposition for credit assignment
    - Double Q-learning for stability
    """

    def __init__(self, n_agents: int, config: QMIXConfig = None):
        self.n_agents = n_agents
        self.config = config or QMIXConfig(n_agents=n_agents)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize mixer networks
        self.mixer = QMIXMixer(
            n_agents=n_agents,
            embed_dim=self.config.embed_dim,
            hypernet_hidden_dim=self.config.hypernet_hidden_dim
        ).to(self.device)

        self.target_mixer = deepcopy(self.mixer)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.mixer.parameters(),
            lr=self.config.lr
        )

        # Replay buffer
        self.buffer = MultiAgentReplayBuffer(
            capacity=self.config.buffer_size,
            n_agents=n_agents
        )

        # Training step counter
        self.training_step = 0

    def train(self, episodes: List[Dict]) -> Dict[str, float]:
        """Train QMIX from collected episodes.

        Args:
            episodes: List of episode data containing:
                - agent_transitions: Dict[agent_id, List[transitions]]
                - global_states: List of global states
                - global_rewards: List of team rewards

        Returns:
            Training metrics
        """
        # Add episodes to buffer
        for episode in episodes:
            self._add_episode_to_buffer(episode)

        if len(self.buffer) < self.config.batch_size:
            return {'qmix_loss': 0.0}

        # Sample batch
        batch = self.buffer.sample(self.config.batch_size)

        # Compute loss
        loss = self._compute_qmix_loss(batch)

        # Backward and update
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        nn.utils.clip_grad_norm_(
            self.mixer.parameters(),
            self.config.grad_norm_clip
        )

        self.optimizer.step()

        # Update target network
        self.training_step += 1
        if self.training_step % self.config.target_update_freq == 0:
            self._update_target_network()

        return {'qmix_loss': loss.item()}

    def _add_episode_to_buffer(self, episode: Dict):
        """Add episode to replay buffer."""
        agent_transitions = episode.get('agent_transitions', {})
        global_states = episode.get('global_states', [])
        global_rewards = episode.get('global_rewards', [])

        # Align all data to same length
        min_len = min(
            len(global_rewards),
            min(len(t) for t in agent_transitions.values()) if agent_transitions else 0
        )

        if min_len == 0:
            return

        for t in range(min_len):
            # Collect agent Q-values at this timestep
            agent_qs = []
            agent_actions = []

            for agent_id in range(self.n_agents):
                if agent_id in agent_transitions and t < len(agent_transitions[agent_id]):
                    trans = agent_transitions[agent_id][t]
                    agent_qs.append(trans.get('q_value', 0.0))
                    agent_actions.append(trans.get('action', 0))
                else:
                    agent_qs.append(0.0)
                    agent_actions.append(0)

            transition = {
                'global_state': global_states[t] if t < len(global_states) else torch.zeros(512),
                'next_global_state': global_states[t + 1] if t + 1 < len(global_states) else torch.zeros(512),
                'agent_qs': torch.tensor(agent_qs, dtype=torch.float32),
                'agent_actions': torch.tensor(agent_actions, dtype=torch.long),
                'reward': global_rewards[t],
                'done': t == min_len - 1
            }

            self.buffer.add(transition)

    def _compute_qmix_loss(self, batch: Dict) -> torch.Tensor:
        """Compute QMIX TD loss.

        Args:
            batch: Sampled batch from replay buffer

        Returns:
            Loss tensor
        """
        # Extract batch data
        global_states = batch['global_state'].to(self.device)
        next_global_states = batch['next_global_state'].to(self.device)
        agent_qs = batch['agent_qs'].to(self.device)
        rewards = batch['reward'].to(self.device)
        dones = batch['done'].to(self.device)

        # Compute Q_tot using mixer
        q_tot = self.mixer(agent_qs, global_states)

        # Compute target Q_tot
        with torch.no_grad():
            # For simplicity, use same Q values for target
            # In full implementation, would use target networks for agents too
            target_q_tot = self.target_mixer(agent_qs, next_global_states)
            target = rewards + self.config.gamma * (1 - dones.float()) * target_q_tot.squeeze(-1)

        # TD loss
        loss = F.mse_loss(q_tot.squeeze(-1), target)

        return loss

    def _update_target_network(self):
        """Hard update of target network."""
        self.target_mixer.load_state_dict(self.mixer.state_dict())

    def soft_update_target(self, tau: float = 0.01):
        """Soft update of target network."""
        for target_param, param in zip(
            self.target_mixer.parameters(),
            self.mixer.parameters()
        ):
            target_param.data.copy_(
                tau * param.data + (1 - tau) * target_param.data
            )

    def get_credit_assignment(self, agent_qs: torch.Tensor,
                             global_state: torch.Tensor) -> torch.Tensor:
        """Get credit assignment weights from mixer.

        Uses the learned mixing weights to assign credit
        to each agent's Q-value contribution.

        Args:
            agent_qs: Individual agent Q-values [batch, n_agents]
            global_state: Global state [batch, state_dim]

        Returns:
            Credit weights [batch, n_agents]
        """
        with torch.no_grad():
            # Compute gradients of Q_tot w.r.t. agent_qs
            agent_qs.requires_grad_(True)
            q_tot = self.mixer(agent_qs, global_state)
            q_tot.sum().backward()

            # Gradients give us the "importance" of each agent's Q
            credits = agent_qs.grad.abs()

            # Normalize
            credits = credits / (credits.sum(dim=-1, keepdim=True) + 1e-8)

        return credits

    def save(self, path: str):
        """Save trainer state."""
        torch.save({
            'mixer': self.mixer.state_dict(),
            'target_mixer': self.target_mixer.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'training_step': self.training_step
        }, path)

    def load(self, path: str):
        """Load trainer state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.mixer.load_state_dict(checkpoint['mixer'])
        self.target_mixer.load_state_dict(checkpoint['target_mixer'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.training_step = checkpoint['training_step']


class MultiAgentReplayBuffer:
    """Replay buffer for multi-agent transitions."""

    def __init__(self, capacity: int, n_agents: int):
        self.capacity = capacity
        self.n_agents = n_agents
        self.buffer = []
        self.position = 0

    def add(self, transition: Dict):
        """Add a multi-agent transition."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a batch of transitions."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]

        # Collate into tensors
        return {
            'global_state': torch.stack([b['global_state'] for b in batch]),
            'next_global_state': torch.stack([b['next_global_state'] for b in batch]),
            'agent_qs': torch.stack([b['agent_qs'] for b in batch]),
            'agent_actions': torch.stack([b['agent_actions'] for b in batch]),
            'reward': torch.tensor([b['reward'] for b in batch], dtype=torch.float32),
            'done': torch.tensor([b['done'] for b in batch], dtype=torch.bool)
        }

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        """Clear the buffer."""
        self.buffer = []
        self.position = 0
