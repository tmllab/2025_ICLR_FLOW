"""Reward Calculation for MARL-Flow"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class RewardConfig:
    """Configuration for reward calculation."""
    # Task-level rewards
    task_success_reward: float = 1.0
    task_failure_penalty: float = -0.5
    retry_penalty: float = -0.1
    efficiency_bonus_factor: float = 0.1

    # Workflow-level rewards
    workflow_completion_bonus: float = 5.0
    partial_completion_factor: float = 0.5
    time_penalty_factor: float = 0.01

    # Collaboration rewards
    help_success_bonus: float = 0.3
    help_failure_penalty: float = -0.1
    downstream_success_bonus: float = 0.2

    # Quality rewards
    validation_pass_bonus: float = 0.2
    first_try_success_bonus: float = 0.3


class RewardCalculator:
    """Calculate rewards for MARL-Flow agents.

    Handles multi-level reward computation:
    1. Task-level: Individual task execution rewards
    2. Workflow-level: Overall workflow completion rewards
    3. Collaboration: Rewards for helping/being helped
    4. Quality: Rewards for validation and efficiency
    """

    def __init__(self, config: RewardConfig = None):
        self.config = config or RewardConfig()

    def compute_task_reward(self, task_result: Dict[str, Any]) -> float:
        """Compute reward for a single task execution.

        Args:
            task_result: Dictionary containing:
                - status: 'completed' or 'failed'
                - retry_count: Number of retries
                - execution_time: Time taken
                - validation_passed: Whether validation passed
                - first_try_success: Whether succeeded on first try

        Returns:
            Task reward value
        """
        reward = 0.0

        # Base success/failure reward
        if task_result.get('status') == 'completed':
            reward += self.config.task_success_reward

            # First try success bonus
            if task_result.get('first_try_success', False):
                reward += self.config.first_try_success_bonus

            # Validation pass bonus
            if task_result.get('validation_passed', False):
                reward += self.config.validation_pass_bonus

        else:
            reward += self.config.task_failure_penalty

        # Retry penalty
        retry_count = task_result.get('retry_count', 0)
        reward += self.config.retry_penalty * retry_count

        # Efficiency bonus (faster is better)
        execution_time = task_result.get('execution_time', 1.0)
        if execution_time > 0:
            efficiency_bonus = self.config.efficiency_bonus_factor / execution_time
            reward += min(efficiency_bonus, 0.5)  # Cap the bonus

        return reward

    def compute_workflow_reward(self, workflow_result: Dict[str, Any]) -> float:
        """Compute reward for overall workflow completion.

        Args:
            workflow_result: Dictionary containing:
                - total_tasks: Total number of tasks
                - completed_tasks: Number of completed tasks
                - failed_tasks: Number of failed tasks
                - total_time: Total execution time
                - dependency_satisfaction: How well dependencies were met

        Returns:
            Workflow-level reward
        """
        reward = 0.0

        total_tasks = workflow_result.get('total_tasks', 1)
        completed_tasks = workflow_result.get('completed_tasks', 0)
        total_time = workflow_result.get('total_time', 1.0)

        # Completion rate
        completion_rate = completed_tasks / total_tasks if total_tasks > 0 else 0

        # Full completion bonus
        if completion_rate == 1.0:
            reward += self.config.workflow_completion_bonus
        else:
            # Partial completion reward
            reward += self.config.partial_completion_factor * completion_rate * self.config.workflow_completion_bonus

        # Time penalty
        reward -= self.config.time_penalty_factor * total_time

        return reward

    def compute_collaboration_reward(self, collaboration_info: Dict[str, Any]) -> float:
        """Compute reward for collaboration events.

        Args:
            collaboration_info: Dictionary containing:
                - helped_other: Whether agent helped another
                - help_successful: Whether the help was successful
                - received_help: Whether agent received help
                - help_led_to_success: Whether help led to task success

        Returns:
            Collaboration reward
        """
        reward = 0.0

        # Reward for successful help
        if collaboration_info.get('helped_other', False):
            if collaboration_info.get('help_successful', False):
                reward += self.config.help_success_bonus
            else:
                reward += self.config.help_failure_penalty

        # Bonus for downstream task success
        if collaboration_info.get('downstream_success', False):
            reward += self.config.downstream_success_bonus

        return reward

    def compute_total_reward(self, task_result: Optional[Dict] = None,
                            workflow_result: Optional[Dict] = None,
                            collaboration_info: Optional[Dict] = None) -> float:
        """Compute total reward from all components.

        Args:
            task_result: Task execution result
            workflow_result: Workflow completion result
            collaboration_info: Collaboration information

        Returns:
            Total reward
        """
        reward = 0.0

        if task_result is not None:
            reward += self.compute_task_reward(task_result)

        if workflow_result is not None:
            reward += self.compute_workflow_reward(workflow_result)

        if collaboration_info is not None:
            reward += self.compute_collaboration_reward(collaboration_info)

        return reward

    def compute_shaped_reward(self, current_state: Dict, next_state: Dict,
                             action: Dict, potential_func: callable = None) -> float:
        """Compute potential-based reward shaping.

        Uses potential-based shaping: F(s, s') = γ * Φ(s') - Φ(s)
        where Φ is a potential function.

        Args:
            current_state: State before action
            next_state: State after action
            action: Action taken
            potential_func: Custom potential function

        Returns:
            Shaped reward
        """
        gamma = 0.99

        if potential_func is None:
            # Default potential: progress towards completion
            potential_func = self._default_potential

        current_potential = potential_func(current_state)
        next_potential = potential_func(next_state)

        shaped_reward = gamma * next_potential - current_potential

        return shaped_reward

    def _default_potential(self, state: Dict) -> float:
        """Default potential function based on workflow progress.

        Args:
            state: Current state dictionary

        Returns:
            Potential value
        """
        # Progress potential
        tasks = state.get('tasks', {})
        if not tasks:
            return 0.0

        completed = sum(1 for t in tasks.values()
                       if isinstance(t, dict) and t.get('status') == 'completed')
        total = len(tasks)

        progress_potential = completed / total if total > 0 else 0

        # Quality potential (fewer failures is better)
        failed = sum(1 for t in tasks.values()
                    if isinstance(t, dict) and t.get('status') == 'failed')
        quality_potential = 1.0 - (failed / total if total > 0 else 0)

        return progress_potential + 0.5 * quality_potential


class MultiAgentRewardCalculator(RewardCalculator):
    """Extended reward calculator for multi-agent scenarios.

    Handles team rewards and individual contributions.
    """

    def __init__(self, n_agents: int, config: RewardConfig = None):
        super().__init__(config)
        self.n_agents = n_agents

        # Team reward coefficients
        self.team_weight = 0.5  # Weight for team reward
        self.individual_weight = 0.5  # Weight for individual reward

    def compute_team_reward(self, workflow_result: Dict) -> float:
        """Compute shared team reward."""
        return self.compute_workflow_reward(workflow_result)

    def compute_agent_rewards(self, workflow_result: Dict,
                             agent_contributions: Dict[int, List[Dict]]) -> Dict[int, float]:
        """Compute individual agent rewards.

        Args:
            workflow_result: Overall workflow result
            agent_contributions: Dictionary mapping agent_id to list of task results

        Returns:
            Dictionary mapping agent_id to reward
        """
        # Team reward (shared)
        team_reward = self.compute_team_reward(workflow_result)

        # Individual rewards
        individual_rewards = {}
        for agent_id, contributions in agent_contributions.items():
            # Sum of task rewards
            task_reward = sum(
                self.compute_task_reward(task)
                for task in contributions
            )
            individual_rewards[agent_id] = task_reward

        # Combine team and individual rewards
        combined_rewards = {}
        for agent_id in range(self.n_agents):
            individual = individual_rewards.get(agent_id, 0.0)
            combined = (
                self.team_weight * team_reward / self.n_agents +
                self.individual_weight * individual
            )
            combined_rewards[agent_id] = combined

        return combined_rewards

    def compute_counterfactual_reward(self, agent_id: int,
                                      workflow_result: Dict,
                                      counterfactual_result: Dict) -> float:
        """Compute counterfactual reward (what would happen without this agent).

        Used for difference rewards and credit assignment.

        Args:
            agent_id: Agent to evaluate
            workflow_result: Actual workflow result
            counterfactual_result: Result without this agent's contribution

        Returns:
            Difference reward
        """
        actual_reward = self.compute_workflow_reward(workflow_result)
        counterfactual_reward = self.compute_workflow_reward(counterfactual_result)

        return actual_reward - counterfactual_reward
