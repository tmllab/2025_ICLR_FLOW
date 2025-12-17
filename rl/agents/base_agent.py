"""Base Agent Class for MARL-Flow"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path


class BaseAgent(ABC):
    """Abstract base class for all agents in MARL-Flow.

    Provides common functionality for:
    - Network management
    - Training utilities
    - Checkpoint saving/loading
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # To be initialized by subclasses
        self.policy_net = None
        self.value_net = None
        self.optimizer = None

        # Training statistics
        self.training_step = 0
        self.episode_count = 0

    @abstractmethod
    def select_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[Any, torch.Tensor]:
        """Select an action given the current state.

        Args:
            state: Current state tensor
            deterministic: If True, select action greedily

        Returns:
            action: Selected action
            log_prob: Log probability of the action
        """
        pass

    @abstractmethod
    def encode_state(self, *args, **kwargs) -> torch.Tensor:
        """Encode raw observations into state tensor."""
        pass

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """Get value estimate for a state."""
        if self.value_net is None:
            raise NotImplementedError("Value network not initialized")
        return self.value_net(state)

    def to(self, device: torch.device):
        """Move agent to specified device."""
        self.device = device
        if self.policy_net is not None:
            self.policy_net.to(device)
        if self.value_net is not None:
            self.value_net.to(device)
        return self

    def train_mode(self):
        """Set networks to training mode."""
        if self.policy_net is not None:
            self.policy_net.train()
        if self.value_net is not None:
            self.value_net.train()

    def eval_mode(self):
        """Set networks to evaluation mode."""
        if self.policy_net is not None:
            self.policy_net.eval()
        if self.value_net is not None:
            self.value_net.eval()

    def get_parameters(self) -> List[nn.Parameter]:
        """Get all trainable parameters."""
        params = []
        if self.policy_net is not None:
            params.extend(self.policy_net.parameters())
        if self.value_net is not None:
            params.extend(self.value_net.parameters())
        return params

    def save(self, path: Path):
        """Save agent checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'config': self.config,
            'training_step': self.training_step,
            'episode_count': self.episode_count,
        }

        if self.policy_net is not None:
            checkpoint['policy_net'] = self.policy_net.state_dict()
        if self.value_net is not None:
            checkpoint['value_net'] = self.value_net.state_dict()
        if self.optimizer is not None:
            checkpoint['optimizer'] = self.optimizer.state_dict()

        torch.save(checkpoint, path)

    def load(self, path: Path):
        """Load agent checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.config = checkpoint.get('config', self.config)
        self.training_step = checkpoint.get('training_step', 0)
        self.episode_count = checkpoint.get('episode_count', 0)

        if self.policy_net is not None and 'policy_net' in checkpoint:
            self.policy_net.load_state_dict(checkpoint['policy_net'])
        if self.value_net is not None and 'value_net' in checkpoint:
            self.value_net.load_state_dict(checkpoint['value_net'])
        if self.optimizer is not None and 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

    def update(self, trajectories: List[Dict]) -> Dict[str, float]:
        """Update agent from collected trajectories.

        To be implemented by subclasses using specific algorithms.

        Args:
            trajectories: List of trajectory dictionaries

        Returns:
            Dictionary of training metrics
        """
        raise NotImplementedError("Subclasses must implement update()")
