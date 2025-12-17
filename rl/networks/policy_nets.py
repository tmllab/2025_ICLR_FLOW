"""Policy Networks for MARL-Flow"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Tuple, Optional, Dict, Any


class MetaPolicyNetwork(nn.Module):
    """Meta-Agent Policy Network.

    Responsible for:
    1. Workflow structure selection
    2. Task scheduling decisions
    3. Workflow refinement triggering
    """

    def __init__(self, state_dim: int = 512, hidden_dim: int = 256,
                 workflow_action_dim: int = 20, schedule_action_dim: int = 10):
        super().__init__()

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Workflow selection head
        self.workflow_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, workflow_action_dim)
        )

        # Task scheduling head (outputs priority scores)
        self.schedule_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, schedule_action_dim)
        )

        # Refinement decision head (binary)
        self.refine_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            state: State embedding [batch_size, state_dim]

        Returns:
            Dictionary containing action probabilities for each head
        """
        features = self.feature_extractor(state)

        workflow_logits = self.workflow_head(features)
        workflow_probs = F.softmax(workflow_logits, dim=-1)

        schedule_logits = self.schedule_head(features)
        schedule_probs = F.softmax(schedule_logits, dim=-1)

        refine_prob = self.refine_head(features)

        return {
            'workflow_probs': workflow_probs,
            'workflow_logits': workflow_logits,
            'schedule_probs': schedule_probs,
            'schedule_logits': schedule_logits,
            'refine_prob': refine_prob
        }

    def select_workflow(self, state: torch.Tensor, num_candidates: int,
                        deterministic: bool = False) -> Tuple[int, torch.Tensor]:
        """Select a workflow from candidates.

        Args:
            state: State embedding
            num_candidates: Number of available workflow candidates
            deterministic: Whether to select greedily

        Returns:
            Selected index and log probability
        """
        output = self.forward(state)
        probs = output['workflow_probs'][:, :num_candidates]
        probs = probs / probs.sum(dim=-1, keepdim=True)  # Renormalize

        if deterministic:
            action = probs.argmax(dim=-1)
            log_prob = torch.log(probs.gather(1, action.unsqueeze(-1)))
        else:
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.item(), log_prob

    def get_schedule_priorities(self, state: torch.Tensor, num_tasks: int) -> torch.Tensor:
        """Get priority scores for task scheduling.

        Args:
            state: State embedding
            num_tasks: Number of ready tasks

        Returns:
            Priority scores for each task
        """
        output = self.forward(state)
        priorities = output['schedule_probs'][:, :num_tasks]
        return priorities.squeeze(0)

    def should_refine(self, state: torch.Tensor, threshold: float = 0.5) -> bool:
        """Decide whether to trigger workflow refinement."""
        output = self.forward(state)
        refine_prob = output['refine_prob'].item()
        return refine_prob > threshold

    def evaluate_actions(self, states: torch.Tensor, actions: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Evaluate actions for PPO training.

        Returns log_probs and entropy for each action type.
        """
        output = self.forward(states)

        results = {}

        # Workflow action evaluation
        if 'workflow_action' in actions:
            workflow_dist = Categorical(output['workflow_probs'])
            results['workflow_log_prob'] = workflow_dist.log_prob(actions['workflow_action'])
            results['workflow_entropy'] = workflow_dist.entropy()

        # Schedule action evaluation
        if 'schedule_action' in actions:
            schedule_dist = Categorical(output['schedule_probs'])
            results['schedule_log_prob'] = schedule_dist.log_prob(actions['schedule_action'])
            results['schedule_entropy'] = schedule_dist.entropy()

        return results


class ExecutorPolicyNetwork(nn.Module):
    """Executor Agent Policy Network.

    Responsible for:
    1. Execution strategy selection
    2. Retry strategy selection
    3. Help request decisions
    """

    def __init__(self, state_dim: int = 768, hidden_dim: int = 256,
                 num_strategies: int = 5, num_retry_actions: int = 4):
        super().__init__()

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_strategies = num_strategies
        self.num_retry_actions = num_retry_actions

        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Execution strategy head
        self.strategy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_strategies)
        )

        # Retry action head
        self.retry_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_retry_actions)
        )

        # Help request head (binary)
        self.help_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # Speciality embedding (learnable)
        self.speciality_embedding = nn.Embedding(10, hidden_dim)

    def forward(self, state: torch.Tensor,
                speciality_id: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            state: State embedding [batch_size, state_dim]
            speciality_id: Agent speciality index (optional)

        Returns:
            Dictionary containing action probabilities for each head
        """
        features = self.feature_extractor(state)

        # Add speciality bias if provided
        if speciality_id is not None:
            spec_emb = self.speciality_embedding(
                torch.tensor([speciality_id], device=state.device)
            )
            features = features + spec_emb

        strategy_logits = self.strategy_head(features)
        strategy_probs = F.softmax(strategy_logits, dim=-1)

        retry_logits = self.retry_head(features)
        retry_probs = F.softmax(retry_logits, dim=-1)

        help_prob = self.help_head(features)

        return {
            'strategy_probs': strategy_probs,
            'strategy_logits': strategy_logits,
            'retry_probs': retry_probs,
            'retry_logits': retry_logits,
            'help_prob': help_prob
        }

    def select_strategy(self, state: torch.Tensor,
                        deterministic: bool = False) -> Tuple[int, torch.Tensor]:
        """Select execution strategy.

        Args:
            state: State embedding
            deterministic: Whether to select greedily

        Returns:
            Selected strategy index and log probability
        """
        output = self.forward(state)
        probs = output['strategy_probs']

        if deterministic:
            action = probs.argmax(dim=-1)
            log_prob = torch.log(probs.gather(1, action.unsqueeze(-1))).squeeze(-1)
        else:
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.item(), log_prob

    def select_retry_action(self, state: torch.Tensor,
                           deterministic: bool = False) -> Tuple[int, torch.Tensor]:
        """Select retry action.

        Actions: 0=retry, 1=modify_prompt, 2=decompose, 3=give_up
        """
        output = self.forward(state)
        probs = output['retry_probs']

        if deterministic:
            action = probs.argmax(dim=-1)
            log_prob = torch.log(probs.gather(1, action.unsqueeze(-1))).squeeze(-1)
        else:
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.item(), log_prob

    def should_request_help(self, state: torch.Tensor, threshold: float = 0.5) -> bool:
        """Decide whether to request help from other agents."""
        output = self.forward(state)
        help_prob = output['help_prob'].item()
        return help_prob > threshold

    def evaluate_actions(self, states: torch.Tensor,
                        actions: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Evaluate actions for PPO training."""
        output = self.forward(states)

        results = {}

        # Strategy action evaluation
        if 'strategy_action' in actions:
            strategy_dist = Categorical(output['strategy_probs'])
            results['strategy_log_prob'] = strategy_dist.log_prob(actions['strategy_action'])
            results['strategy_entropy'] = strategy_dist.entropy()

        # Retry action evaluation
        if 'retry_action' in actions:
            retry_dist = Categorical(output['retry_probs'])
            results['retry_log_prob'] = retry_dist.log_prob(actions['retry_action'])
            results['retry_entropy'] = retry_dist.entropy()

        return results


class WorkflowGeneratorNetwork(nn.Module):
    """Network for generating workflow DAG structure.

    This network generates workflow structure autoregressively:
    1. Generate nodes (tasks)
    2. Generate edges (dependencies)
    3. Assign agents to tasks
    """

    def __init__(self, state_dim: int = 512, hidden_dim: int = 256,
                 max_nodes: int = 15, max_agents: int = 5):
        super().__init__()

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.max_nodes = max_nodes
        self.max_agents = max_agents

        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Node generation LSTM
        self.node_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        # Action heads
        # Add node action
        self.add_node_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # [continue, stop]
        )

        # Node features head (objective embedding)
        self.node_feature_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)  # Output node embedding
        )

        # Edge prediction head
        self.edge_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # Agent assignment head
        self.agent_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_agents)
        )

    def forward(self, state: torch.Tensor, partial_dag: Optional[Dict] = None):
        """Generate or extend workflow structure."""
        batch_size = state.shape[0]

        # Encode initial state
        h = self.state_encoder(state)

        # Initialize LSTM state
        lstm_h = h.unsqueeze(0)
        lstm_c = torch.zeros_like(lstm_h)

        generated_nodes = []
        node_embeddings = []

        # Generate nodes autoregressively
        for i in range(self.max_nodes):
            # LSTM step
            lstm_input = h.unsqueeze(1)
            lstm_out, (lstm_h, lstm_c) = self.node_lstm(lstm_input, (lstm_h, lstm_c))

            node_h = lstm_out.squeeze(1)

            # Decide whether to add node
            add_logits = self.add_node_head(node_h)
            add_probs = F.softmax(add_logits, dim=-1)

            # Sample add decision
            add_dist = Categorical(add_probs)
            add_action = add_dist.sample()

            if add_action.item() == 1:  # Stop
                break

            # Generate node features
            node_emb = self.node_feature_head(node_h)
            node_embeddings.append(node_emb)

            # Assign agent
            agent_logits = self.agent_head(node_h)
            agent_probs = F.softmax(agent_logits, dim=-1)
            agent_dist = Categorical(agent_probs)
            agent_id = agent_dist.sample()

            generated_nodes.append({
                'embedding': node_emb,
                'agent_id': agent_id.item()
            })

            # Update h for next step
            h = node_emb

        # Generate edges between nodes
        edges = []
        if len(node_embeddings) > 1:
            node_embs = torch.stack(node_embeddings, dim=1)  # [batch, num_nodes, hidden]

            for i in range(len(node_embeddings)):
                for j in range(i + 1, len(node_embeddings)):
                    # Predict edge probability
                    pair_emb = torch.cat([node_embs[:, i], node_embs[:, j]], dim=-1)
                    edge_prob = self.edge_head(pair_emb)

                    if edge_prob.item() > 0.5:
                        edges.append((i, j))

        return {
            'nodes': generated_nodes,
            'edges': edges,
            'node_embeddings': node_embeddings
        }
