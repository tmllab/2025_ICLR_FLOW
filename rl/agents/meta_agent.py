"""Meta-Agent for MARL-Flow

The Meta-Agent is responsible for high-level decisions:
1. Workflow structure selection/generation
2. Task scheduling and prioritization
3. Workflow refinement triggering
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

from rl.agents.base_agent import BaseAgent
from rl.networks.policy_nets import MetaPolicyNetwork
from rl.networks.value_nets import MetaValueNetwork
from rl.config import MetaAgentConfig


class MetaAgent(BaseAgent):
    """Meta-Agent: Central controller for workflow orchestration.

    Implements the high-level decision making in the MARL-Flow system,
    including workflow selection, task scheduling, and refinement control.
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

        # Use default config if not provided
        self.agent_config = MetaAgentConfig()
        if config:
            for k, v in config.items():
                if hasattr(self.agent_config, k):
                    setattr(self.agent_config, k, v)

        # Initialize networks
        self.policy_net = MetaPolicyNetwork(
            state_dim=self.agent_config.state_dim,
            workflow_action_dim=self.agent_config.workflow_action_dim,
            schedule_action_dim=self.agent_config.schedule_action_dim
        ).to(self.device)

        self.value_net = MetaValueNetwork(
            state_dim=self.agent_config.state_dim
        ).to(self.device)

        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            list(self.policy_net.parameters()) + list(self.value_net.parameters()),
            lr=self.config.get('lr', 3e-4)
        )

        # Text encoder for task descriptions (simple version)
        self.text_encoder = self._create_text_encoder()

    def _create_text_encoder(self) -> nn.Module:
        """Create a simple text encoder.

        In production, replace with BERT or similar.
        """
        return nn.Sequential(
            nn.Linear(768, 512),  # Assuming pre-computed embeddings
            nn.ReLU(),
            nn.Linear(512, self.agent_config.state_dim)
        ).to(self.device)

    def encode_state(self, task_description: str, workflow_state: Dict,
                     history_stats: Optional[Dict] = None) -> torch.Tensor:
        """Encode the current state for decision making.

        Args:
            task_description: The overall task description
            workflow_state: Current workflow state (tasks, statuses, etc.)
            history_stats: Historical execution statistics

        Returns:
            State tensor [1, state_dim]
        """
        # For now, use a simple encoding
        # In production, use proper text encoder

        # Encode task description (placeholder - use real encoder)
        task_features = self._encode_text(task_description)

        # Encode workflow state
        workflow_features = self._encode_workflow(workflow_state)

        # Encode history stats
        if history_stats:
            history_features = self._encode_history(history_stats)
        else:
            history_features = torch.zeros(128, device=self.device)

        # Concatenate and project
        combined = torch.cat([task_features, workflow_features, history_features])
        state = combined.unsqueeze(0)  # Add batch dimension

        return state

    def _encode_text(self, text: str) -> torch.Tensor:
        """Encode text to embedding."""
        # Placeholder: In production, use BERT or similar
        # For now, return random features
        return torch.randn(256, device=self.device)

    def _encode_workflow(self, workflow_state: Dict) -> torch.Tensor:
        """Encode workflow state to embedding."""
        # Extract key features
        if not workflow_state:
            return torch.zeros(128, device=self.device)

        tasks = workflow_state.get('tasks', {})
        n_tasks = len(tasks)
        n_completed = sum(1 for t in tasks.values()
                         if isinstance(t, dict) and t.get('status') == 'completed')
        n_pending = sum(1 for t in tasks.values()
                       if isinstance(t, dict) and t.get('status') == 'pending')
        n_failed = sum(1 for t in tasks.values()
                      if isinstance(t, dict) and t.get('status') == 'failed')

        # Create feature vector
        features = torch.tensor([
            n_tasks / 20.0,  # Normalize
            n_completed / max(n_tasks, 1),
            n_pending / max(n_tasks, 1),
            n_failed / max(n_tasks, 1),
        ], device=self.device)

        # Pad to fixed size
        padded = torch.zeros(128, device=self.device)
        padded[:len(features)] = features

        return padded

    def _encode_history(self, history_stats: Dict) -> torch.Tensor:
        """Encode historical execution statistics."""
        features = torch.tensor([
            history_stats.get('avg_success_rate', 0.5),
            history_stats.get('avg_execution_time', 0.5),
            history_stats.get('avg_retries', 0.0) / 5.0,
        ], device=self.device)

        padded = torch.zeros(128, device=self.device)
        padded[:len(features)] = features

        return padded

    def select_action(self, state: torch.Tensor,
                     deterministic: bool = False) -> Tuple[Dict, torch.Tensor]:
        """Select high-level action.

        Args:
            state: Current state tensor
            deterministic: Whether to select greedily

        Returns:
            action_dict: Dictionary of actions
            log_prob: Combined log probability
        """
        self.eval_mode()

        with torch.no_grad():
            output = self.policy_net(state)

        # Sample actions from each head
        from torch.distributions import Categorical

        # Workflow selection
        workflow_dist = Categorical(output['workflow_probs'])
        if deterministic:
            workflow_action = output['workflow_probs'].argmax(dim=-1)
        else:
            workflow_action = workflow_dist.sample()
        workflow_log_prob = workflow_dist.log_prob(workflow_action)

        # Schedule selection
        schedule_dist = Categorical(output['schedule_probs'])
        if deterministic:
            schedule_action = output['schedule_probs'].argmax(dim=-1)
        else:
            schedule_action = schedule_dist.sample()
        schedule_log_prob = schedule_dist.log_prob(schedule_action)

        # Refine decision
        refine_prob = output['refine_prob'].item()
        refine_action = refine_prob > self.agent_config.refine_threshold

        action_dict = {
            'workflow_idx': workflow_action.item(),
            'schedule_priority': schedule_action.item(),
            'should_refine': refine_action
        }

        # Combined log prob (sum of independent actions)
        total_log_prob = workflow_log_prob + schedule_log_prob

        return action_dict, total_log_prob

    def select_workflow(self, candidates: List, objective: str) -> Any:
        """Select the best workflow from candidates.

        Args:
            candidates: List of workflow candidates
            objective: Task objective description

        Returns:
            Selected workflow
        """
        if not candidates:
            raise ValueError("No workflow candidates provided")

        # Encode state
        state = self.encode_state(objective, {}, None)

        # Get selection
        action, _ = self.select_action(state, deterministic=True)
        idx = action['workflow_idx'] % len(candidates)

        return candidates[idx]

    def prioritize_tasks(self, ready_tasks: List, workflow_state: Dict) -> List:
        """Prioritize ready tasks for execution.

        Args:
            ready_tasks: List of tasks ready to execute
            workflow_state: Current workflow state

        Returns:
            Sorted list of tasks by priority
        """
        if not ready_tasks:
            return []

        # Encode state for each task
        state = self.encode_state("", workflow_state, None)

        # Get priorities from policy
        with torch.no_grad():
            output = self.policy_net(state)
            priorities = output['schedule_probs'].squeeze(0).cpu().numpy()

        # Map priorities to tasks
        n_tasks = min(len(ready_tasks), len(priorities))
        task_priorities = list(zip(ready_tasks[:n_tasks], priorities[:n_tasks]))

        # Sort by priority (descending)
        task_priorities.sort(key=lambda x: x[1], reverse=True)

        return [t for t, _ in task_priorities]

    def should_refine_workflow(self, workflow_state: Dict) -> bool:
        """Decide whether to trigger workflow refinement.

        Args:
            workflow_state: Current workflow state

        Returns:
            Boolean indicating whether to refine
        """
        state = self.encode_state("", workflow_state, None)

        with torch.no_grad():
            output = self.policy_net(state)
            refine_prob = output['refine_prob'].item()

        return refine_prob > self.agent_config.refine_threshold

    def update(self, trajectories: List[Dict]) -> Dict[str, float]:
        """Update agent using PPO.

        Args:
            trajectories: List of trajectory data

        Returns:
            Training metrics
        """
        self.train_mode()

        # Prepare batch data
        states = torch.stack([t['state'] for t in trajectories]).to(self.device)
        actions = {
            'workflow_action': torch.tensor([t['action']['workflow_idx'] for t in trajectories]),
            'schedule_action': torch.tensor([t['action']['schedule_priority'] for t in trajectories])
        }
        old_log_probs = torch.stack([t['log_prob'] for t in trajectories]).to(self.device)
        returns = torch.tensor([t['return'] for t in trajectories], dtype=torch.float32).to(self.device)
        advantages = torch.tensor([t['advantage'] for t in trajectories], dtype=torch.float32).to(self.device)

        # PPO update
        clip_ratio = self.config.get('clip_ratio', 0.2)

        for _ in range(self.config.get('ppo_epochs', 4)):
            # Forward pass
            output = self.policy_net(states)
            values = self.value_net(states).squeeze(-1)

            # Evaluate actions
            eval_results = self.policy_net.evaluate_actions(states, actions)
            new_log_probs = eval_results['workflow_log_prob'] + eval_results['schedule_log_prob']

            # Policy loss (PPO-clip)
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = nn.functional.mse_loss(values, returns)

            # Entropy bonus
            entropy = eval_results.get('workflow_entropy', torch.tensor(0.0)).mean()
            entropy_loss = -self.config.get('entropy_coef', 0.01) * entropy

            # Total loss
            loss = policy_loss + 0.5 * value_loss + entropy_loss

            # Backward and update
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.get_parameters(), self.config.get('max_grad_norm', 0.5))
            self.optimizer.step()

        self.training_step += 1

        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item() if torch.is_tensor(entropy) else entropy
        }
