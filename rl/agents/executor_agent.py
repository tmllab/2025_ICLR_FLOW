"""Executor Agent for MARL-Flow

The Executor Agent is responsible for low-level task execution decisions:
1. Execution strategy selection
2. Retry strategy selection
3. Help request decisions
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

from rl.agents.base_agent import BaseAgent
from rl.networks.policy_nets import ExecutorPolicyNetwork
from rl.networks.value_nets import ExecutorValueNetwork
from rl.config import ExecutorAgentConfig


class ExecutorAgent(BaseAgent):
    """Executor Agent: Task execution specialist.

    Each Executor Agent can have a speciality (e.g., code generation,
    UI design) and learns to execute tasks within its domain.
    """

    # Strategy definitions
    STRATEGIES = [
        {'name': 'concise', 'detail_level': 'low', 'prompt_style': 'brief'},
        {'name': 'detailed', 'detail_level': 'high', 'prompt_style': 'comprehensive'},
        {'name': 'step_by_step', 'detail_level': 'medium', 'prompt_style': 'structured'},
        {'name': 'example_based', 'detail_level': 'medium', 'prompt_style': 'examples'},
        {'name': 'formal', 'detail_level': 'high', 'prompt_style': 'academic'},
    ]

    # Retry actions
    RETRY_ACTIONS = ['retry', 'modify_prompt', 'decompose', 'give_up']

    def __init__(self, agent_id: int = 0, speciality: str = None,
                 config: Dict[str, Any] = None):
        super().__init__(config)

        self.agent_id = agent_id
        self.speciality = speciality

        # Use default config if not provided
        self.agent_config = ExecutorAgentConfig()
        if config:
            for k, v in config.items():
                if hasattr(self.agent_config, k):
                    setattr(self.agent_config, k, v)

        # Get speciality index
        self.speciality_idx = self._get_speciality_index(speciality)

        # Initialize networks
        self.policy_net = ExecutorPolicyNetwork(
            state_dim=self.agent_config.state_dim,
            num_strategies=self.agent_config.num_strategies,
            num_retry_actions=self.agent_config.num_retry_actions
        ).to(self.device)

        self.value_net = ExecutorValueNetwork(
            state_dim=self.agent_config.state_dim
        ).to(self.device)

        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            list(self.policy_net.parameters()) + list(self.value_net.parameters()),
            lr=self.config.get('lr', 3e-4)
        )

        # Execution history (for learning from experience)
        self.execution_history = []

    def _get_speciality_index(self, speciality: str) -> int:
        """Map speciality string to index."""
        if speciality is None:
            return 0

        specialities = self.agent_config.specialities
        if speciality in specialities:
            return specialities.index(speciality)
        return 0

    def encode_state(self, task_objective: str, context: str,
                     next_objective: str, execution_history: Optional[List] = None,
                     validation_feedback: Optional[str] = None) -> torch.Tensor:
        """Encode execution state for decision making.

        Args:
            task_objective: Current task objective
            context: Context from upstream tasks
            next_objective: Downstream task objectives
            execution_history: Previous execution attempts
            validation_feedback: Latest validation feedback

        Returns:
            State tensor [1, state_dim]
        """
        # Encode task objective
        task_features = self._encode_text(task_objective)

        # Encode context
        context_features = self._encode_text(context) if context else torch.zeros(256, device=self.device)

        # Encode downstream objectives
        next_features = self._encode_text(next_objective) if next_objective else torch.zeros(256, device=self.device)

        # Encode execution history
        if execution_history:
            history_features = self._encode_history(execution_history)
        else:
            history_features = torch.zeros(128, device=self.device)

        # Encode validation feedback
        if validation_feedback:
            feedback_features = self._encode_text(validation_feedback)[:128]
        else:
            feedback_features = torch.zeros(128, device=self.device)

        # Combine all features
        combined = torch.cat([
            task_features,
            context_features,
            next_features[:128],  # Truncate
            history_features,
            feedback_features
        ])

        # Pad or truncate to state_dim
        state = torch.zeros(self.agent_config.state_dim, device=self.device)
        state[:min(len(combined), self.agent_config.state_dim)] = combined[:self.agent_config.state_dim]

        return state.unsqueeze(0)  # Add batch dimension

    def _encode_text(self, text: str) -> torch.Tensor:
        """Encode text to embedding (placeholder)."""
        # In production, use BERT or similar
        return torch.randn(256, device=self.device)

    def _encode_history(self, history: List) -> torch.Tensor:
        """Encode execution history."""
        # Extract features from history
        n_attempts = len(history)
        success_count = sum(1 for h in history if h.get('status') == 'completed')

        features = torch.tensor([
            n_attempts / 10.0,  # Normalize
            success_count / max(n_attempts, 1),
        ], device=self.device)

        # Pad to fixed size
        padded = torch.zeros(128, device=self.device)
        padded[:len(features)] = features

        return padded

    def select_action(self, state: torch.Tensor,
                     deterministic: bool = False) -> Tuple[Dict, torch.Tensor]:
        """Select execution action.

        Args:
            state: Current state tensor
            deterministic: Whether to select greedily

        Returns:
            action_dict: Dictionary of actions
            log_prob: Log probability
        """
        self.eval_mode()

        with torch.no_grad():
            output = self.policy_net(state, self.speciality_idx)

        from torch.distributions import Categorical

        # Select strategy
        strategy_dist = Categorical(output['strategy_probs'])
        if deterministic:
            strategy_action = output['strategy_probs'].argmax(dim=-1)
        else:
            strategy_action = strategy_dist.sample()
        strategy_log_prob = strategy_dist.log_prob(strategy_action)

        action_dict = {
            'strategy_idx': strategy_action.item(),
            'strategy': self.STRATEGIES[strategy_action.item()]
        }

        return action_dict, strategy_log_prob

    def select_strategy(self, task_objective: str, context: str,
                        next_objective: str) -> Dict[str, Any]:
        """Select execution strategy for a task.

        Args:
            task_objective: Task to execute
            context: Context from upstream
            next_objective: Downstream needs

        Returns:
            Strategy configuration dictionary
        """
        state = self.encode_state(task_objective, context, next_objective)
        action, _ = self.select_action(state, deterministic=False)
        return action['strategy']

    def select_retry_action(self, task_objective: str, result: str,
                           feedback: str, iteration: int) -> str:
        """Select retry strategy after validation failure.

        Args:
            task_objective: Original task
            result: Current result
            feedback: Validation feedback
            iteration: Current retry iteration

        Returns:
            Retry action string
        """
        state = self.encode_state(
            task_objective=task_objective,
            context=result,
            next_objective="",
            execution_history=[{'iteration': iteration}],
            validation_feedback=feedback
        )

        with torch.no_grad():
            output = self.policy_net(state, self.speciality_idx)

        from torch.distributions import Categorical
        retry_dist = Categorical(output['retry_probs'])
        retry_action = retry_dist.sample()

        return self.RETRY_ACTIONS[retry_action.item()]

    def should_request_help(self, task_objective: str, result: str,
                           feedback: str) -> Tuple[bool, Optional[str]]:
        """Decide whether to request help from other agents.

        Args:
            task_objective: Current task
            result: Current result
            feedback: Validation feedback

        Returns:
            Tuple of (should_request, help_type)
        """
        state = self.encode_state(
            task_objective=task_objective,
            context=result,
            next_objective="",
            validation_feedback=feedback
        )

        with torch.no_grad():
            output = self.policy_net(state, self.speciality_idx)
            help_prob = output['help_prob'].item()

        if help_prob > 0.5:
            # Determine help type based on speciality
            if self.speciality == 'code_generation':
                help_type = 'review'
            elif self.speciality == 'integration':
                help_type = 'component'
            else:
                help_type = 'general'
            return True, help_type

        return False, None

    def record_execution(self, task_objective: str, strategy: Dict,
                        result: str, success: bool, execution_time: float):
        """Record execution for learning.

        Args:
            task_objective: Executed task
            strategy: Strategy used
            result: Execution result
            success: Whether successful
            execution_time: Time taken
        """
        self.execution_history.append({
            'task': task_objective,
            'strategy': strategy,
            'success': success,
            'execution_time': execution_time,
            'speciality': self.speciality
        })

        # Keep history bounded
        if len(self.execution_history) > 1000:
            self.execution_history = self.execution_history[-1000:]

    def update(self, trajectories: List[Dict]) -> Dict[str, float]:
        """Update agent using PPO.

        Args:
            trajectories: List of trajectory data

        Returns:
            Training metrics
        """
        self.train_mode()

        if not trajectories:
            return {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0}

        # Prepare batch data
        states = torch.stack([t['state'] for t in trajectories]).to(self.device)
        actions = {
            'strategy_action': torch.tensor([t['action']['strategy_idx'] for t in trajectories])
        }
        old_log_probs = torch.stack([t['log_prob'] for t in trajectories]).to(self.device)
        returns = torch.tensor([t['return'] for t in trajectories], dtype=torch.float32).to(self.device)
        advantages = torch.tensor([t['advantage'] for t in trajectories], dtype=torch.float32).to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        clip_ratio = self.config.get('clip_ratio', 0.2)
        policy_losses = []
        value_losses = []
        entropies = []

        for _ in range(self.config.get('ppo_epochs', 4)):
            # Forward pass
            output = self.policy_net(states, self.speciality_idx)
            values = self.value_net(states, self.speciality_idx).squeeze(-1)

            # Evaluate strategy action
            from torch.distributions import Categorical
            strategy_dist = Categorical(output['strategy_probs'])
            new_log_probs = strategy_dist.log_prob(actions['strategy_action'].to(self.device))
            entropy = strategy_dist.entropy().mean()

            # Policy loss (PPO-clip)
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = nn.functional.mse_loss(values, returns)

            # Entropy bonus
            entropy_loss = -self.config.get('entropy_coef', 0.01) * entropy

            # Total loss
            loss = policy_loss + 0.5 * value_loss + entropy_loss

            # Backward and update
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.get_parameters(), self.config.get('max_grad_norm', 0.5))
            self.optimizer.step()

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropies.append(entropy.item())

        self.training_step += 1

        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy': np.mean(entropies)
        }

    def get_speciality_match_score(self, task_type: str) -> float:
        """Get how well this agent's speciality matches a task type.

        Args:
            task_type: Type of task (inferred or explicit)

        Returns:
            Match score between 0 and 1
        """
        speciality_task_map = {
            'code_generation': ['code', 'implement', 'function', 'algorithm', 'develop'],
            'ui_design': ['ui', 'interface', 'design', 'layout', 'frontend'],
            'system_architecture': ['architecture', 'design', 'system', 'structure'],
            'integration': ['integrate', 'connect', 'combine', 'merge'],
            'data_processing': ['data', 'process', 'analyze', 'transform']
        }

        if self.speciality not in speciality_task_map:
            return 0.5  # Default score for unknown speciality

        keywords = speciality_task_map[self.speciality]
        task_lower = task_type.lower()

        matches = sum(1 for kw in keywords if kw in task_lower)
        return min(1.0, matches / len(keywords) + 0.3)  # Base score of 0.3
