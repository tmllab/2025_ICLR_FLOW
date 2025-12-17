"""Replay Buffers for RL Training"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import deque
import random


class ReplayBuffer:
    """Standard experience replay buffer."""

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add(self, transition: Dict[str, Any]):
        """Add a transition to the buffer."""
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> List[Dict]:
        """Sample a batch of transitions."""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def sample_tensors(self, batch_size: int, device: torch.device = None) -> Dict[str, torch.Tensor]:
        """Sample and return as tensors."""
        batch = self.sample(batch_size)

        if not batch:
            return {}

        device = device or torch.device('cpu')

        result = {}
        keys = batch[0].keys()

        for key in keys:
            values = [b[key] for b in batch]

            if isinstance(values[0], torch.Tensor):
                result[key] = torch.stack(values).to(device)
            elif isinstance(values[0], (int, float)):
                result[key] = torch.tensor(values, device=device)
            elif isinstance(values[0], bool):
                result[key] = torch.tensor(values, dtype=torch.bool, device=device)
            else:
                result[key] = values

        return result

    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        return self.buffer[idx]


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer.

    Samples transitions with probability proportional to their TD error.
    """

    def __init__(self, capacity: int = 10000, alpha: float = 0.6,
                 beta: float = 0.4, beta_increment: float = 0.001):
        self.capacity = capacity
        self.alpha = alpha  # Prioritization exponent
        self.beta = beta  # Importance sampling exponent
        self.beta_increment = beta_increment

        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0

    def add(self, transition: Dict[str, Any], priority: float = None):
        """Add transition with optional priority."""
        if priority is None:
            priority = self.max_priority

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition

        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[List[Dict], np.ndarray, np.ndarray]:
        """Sample with priorities.

        Returns:
            transitions: Sampled transitions
            indices: Indices of sampled transitions
            weights: Importance sampling weights
        """
        n = len(self.buffer)

        # Compute sampling probabilities
        priorities = self.priorities[:n]
        probs = priorities ** self.alpha
        probs /= probs.sum()

        # Sample indices
        indices = np.random.choice(n, batch_size, p=probs, replace=False)

        # Compute importance sampling weights
        weights = (n * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)

        transitions = [self.buffer[i] for i in indices]

        return transitions, indices, weights

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for sampled transitions."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return len(self.buffer)


class EpisodeBuffer:
    """Buffer for storing complete episodes."""

    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.episodes = deque(maxlen=capacity)
        self.current_episode = []

    def add_transition(self, transition: Dict[str, Any]):
        """Add transition to current episode."""
        self.current_episode.append(transition)

    def end_episode(self):
        """Mark current episode as complete and store it."""
        if self.current_episode:
            self.episodes.append(self.current_episode)
            self.current_episode = []

    def sample_episodes(self, n_episodes: int) -> List[List[Dict]]:
        """Sample complete episodes."""
        return random.sample(
            list(self.episodes),
            min(n_episodes, len(self.episodes))
        )

    def sample_transitions(self, batch_size: int) -> List[Dict]:
        """Sample individual transitions from all episodes."""
        all_transitions = [t for ep in self.episodes for t in ep]
        return random.sample(
            all_transitions,
            min(batch_size, len(all_transitions))
        )

    def clear(self):
        """Clear all stored episodes."""
        self.episodes.clear()
        self.current_episode = []

    def __len__(self):
        return len(self.episodes)

    def total_transitions(self):
        """Total number of transitions stored."""
        return sum(len(ep) for ep in self.episodes)
