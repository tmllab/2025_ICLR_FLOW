"""Value Networks for MARL-Flow"""

import torch
import torch.nn as nn
from typing import Optional


class MetaValueNetwork(nn.Module):
    """Value Network for Meta-Agent.

    Estimates the expected return from a given state for
    high-level workflow and scheduling decisions.
    """

    def __init__(self, state_dim: int = 512, hidden_dim: int = 256):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: State embedding [batch_size, state_dim]

        Returns:
            Value estimate [batch_size, 1]
        """
        return self.network(state)


class ExecutorValueNetwork(nn.Module):
    """Value Network for Executor Agent.

    Estimates the expected return from a given state for
    task execution decisions.
    """

    def __init__(self, state_dim: int = 768, hidden_dim: int = 256):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Optional speciality conditioning
        self.speciality_embedding = nn.Embedding(10, hidden_dim)

    def forward(self, state: torch.Tensor,
                speciality_id: Optional[int] = None) -> torch.Tensor:
        """
        Args:
            state: State embedding [batch_size, state_dim]
            speciality_id: Agent speciality index (optional)

        Returns:
            Value estimate [batch_size, 1]
        """
        if speciality_id is not None:
            spec_emb = self.speciality_embedding(
                torch.tensor([speciality_id], device=state.device)
            )
            # Simple additive conditioning
            state = state + spec_emb.expand(state.shape[0], -1)[:, :state.shape[1]]

        return self.network(state)


class QMIXMixer(nn.Module):
    """QMIX Mixing Network for credit assignment.

    Combines individual agent Q-values into a global Q-value
    using a monotonic mixing network.
    """

    def __init__(self, n_agents: int, state_dim: int = 512,
                 embed_dim: int = 64, hypernet_hidden_dim: int = 64):
        super().__init__()

        self.n_agents = n_agents
        self.state_dim = state_dim
        self.embed_dim = embed_dim

        # Hypernetworks for mixing weights
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hypernet_hidden_dim),
            nn.ReLU(),
            nn.Linear(hypernet_hidden_dim, n_agents * embed_dim)
        )

        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hypernet_hidden_dim),
            nn.ReLU(),
            nn.Linear(hypernet_hidden_dim, embed_dim)
        )

        # Hypernetworks for biases
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """
        Mix agent Q-values using state-conditioned weights.

        Args:
            agent_qs: Individual agent Q-values [batch_size, n_agents]
            states: Global state [batch_size, state_dim]

        Returns:
            Mixed global Q-value [batch_size, 1]
        """
        batch_size = agent_qs.shape[0]

        # Reshape agent_qs for mixing
        agent_qs = agent_qs.view(batch_size, 1, self.n_agents)

        # First layer weights (positive via abs)
        w1 = torch.abs(self.hyper_w1(states))
        w1 = w1.view(batch_size, self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(states).view(batch_size, 1, self.embed_dim)

        # First layer forward
        hidden = torch.bmm(agent_qs, w1) + b1
        hidden = torch.relu(hidden)

        # Second layer weights (positive via abs)
        w2 = torch.abs(self.hyper_w2(states))
        w2 = w2.view(batch_size, self.embed_dim, 1)
        b2 = self.hyper_b2(states).view(batch_size, 1, 1)

        # Second layer forward
        q_tot = torch.bmm(hidden, w2) + b2
        q_tot = q_tot.view(batch_size, 1)

        return q_tot


class VDNMixer(nn.Module):
    """Value Decomposition Network (VDN) Mixer.

    Simple additive mixing: Q_tot = sum(Q_i)
    Simpler baseline for QMIX.
    """

    def __init__(self, n_agents: int):
        super().__init__()
        self.n_agents = n_agents

    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            agent_qs: Individual agent Q-values [batch_size, n_agents]
            states: Global state (unused, for API compatibility)

        Returns:
            Summed global Q-value [batch_size, 1]
        """
        return agent_qs.sum(dim=-1, keepdim=True)


class AttentionMixer(nn.Module):
    """Attention-based Mixer for credit assignment.

    Uses attention mechanism to weight agent contributions
    based on the global state.
    """

    def __init__(self, n_agents: int, state_dim: int = 512,
                 hidden_dim: int = 64, num_heads: int = 4):
        super().__init__()

        self.n_agents = n_agents
        self.hidden_dim = hidden_dim

        # Project state to query
        self.state_proj = nn.Linear(state_dim, hidden_dim)

        # Project agent Q-values to keys and values
        self.q_key_proj = nn.Linear(1, hidden_dim)
        self.q_value_proj = nn.Linear(1, hidden_dim)

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            agent_qs: Individual agent Q-values [batch_size, n_agents]
            states: Global state [batch_size, state_dim]

        Returns:
            Attention-weighted global Q-value [batch_size, 1]
        """
        batch_size = agent_qs.shape[0]

        # Project state to query
        query = self.state_proj(states).unsqueeze(1)  # [batch, 1, hidden]

        # Project Q-values to keys and values
        agent_qs = agent_qs.unsqueeze(-1)  # [batch, n_agents, 1]
        keys = self.q_key_proj(agent_qs)    # [batch, n_agents, hidden]
        values = self.q_value_proj(agent_qs)  # [batch, n_agents, hidden]

        # Attention
        attn_output, _ = self.attention(query, keys, values)

        # Output projection
        q_tot = self.output_proj(attn_output.squeeze(1))

        return q_tot
