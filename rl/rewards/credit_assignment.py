"""Credit Assignment for Multi-Agent Systems"""

import torch
import numpy as np
from typing import Dict, List, Any, Tuple
from itertools import combinations


class CreditAssigner:
    """Assign credit to individual agents in multi-agent systems.

    Implements several credit assignment methods:
    1. Uniform: Equal credit to all agents
    2. Proportional: Credit based on task contributions
    3. Shapley: Game-theoretic fair assignment
    4. QMIX-based: Learned credit assignment
    """

    def __init__(self, n_agents: int, method: str = 'proportional'):
        self.n_agents = n_agents
        self.method = method

    def assign_credit(self, global_reward: float,
                     agent_contributions: Dict[int, List[Dict]],
                     workflow_state: Dict = None) -> Dict[int, float]:
        """Assign credit to each agent.

        Args:
            global_reward: Total workflow reward
            agent_contributions: Each agent's task contributions
            workflow_state: Current workflow state (for context)

        Returns:
            Dictionary mapping agent_id to credit
        """
        if self.method == 'uniform':
            return self.uniform_assignment(global_reward)
        elif self.method == 'proportional':
            return self.proportional_assignment(global_reward, agent_contributions)
        elif self.method == 'shapley':
            return self.shapley_assignment(global_reward, agent_contributions, workflow_state)
        else:
            raise ValueError(f"Unknown credit assignment method: {self.method}")

    def uniform_assignment(self, global_reward: float) -> Dict[int, float]:
        """Assign equal credit to all agents."""
        per_agent = global_reward / self.n_agents
        return {i: per_agent for i in range(self.n_agents)}

    def proportional_assignment(self, global_reward: float,
                               agent_contributions: Dict[int, List[Dict]]) -> Dict[int, float]:
        """Assign credit proportional to contributions.

        Considers:
        - Number of tasks completed
        - Task success rate
        - Task complexity (proxy: execution time)
        """
        scores = {}
        total_score = 0.0

        for agent_id in range(self.n_agents):
            contributions = agent_contributions.get(agent_id, [])

            if not contributions:
                scores[agent_id] = 0.1  # Small base score
                continue

            # Count successful tasks
            successes = sum(1 for c in contributions
                          if c.get('status') == 'completed')

            # Average success rate
            success_rate = successes / len(contributions) if contributions else 0

            # Complexity proxy (average execution time)
            avg_time = np.mean([c.get('execution_time', 1.0) for c in contributions])

            # Compute score
            score = (
                0.5 * successes +  # Number of successes
                0.3 * success_rate * len(contributions) +  # Success rate
                0.2 * (1.0 / (avg_time + 0.1))  # Efficiency
            )
            scores[agent_id] = max(score, 0.1)  # Minimum score

        total_score = sum(scores.values())

        # Normalize to distribute global reward
        credits = {
            agent_id: (score / total_score) * global_reward
            for agent_id, score in scores.items()
        }

        return credits

    def shapley_assignment(self, global_reward: float,
                          agent_contributions: Dict[int, List[Dict]],
                          workflow_state: Dict) -> Dict[int, float]:
        """Compute Shapley values for credit assignment.

        Shapley value: Average marginal contribution of an agent
        across all possible coalitions.

        Note: Exponential complexity O(2^n), only feasible for small n.
        """
        if self.n_agents > 8:
            # Fall back to proportional for large n
            return self.proportional_assignment(global_reward, agent_contributions)

        agents = list(range(self.n_agents))
        shapley_values = {i: 0.0 for i in agents}

        # For each agent, compute marginal contributions
        for agent in agents:
            marginal_sum = 0.0
            count = 0

            # Consider all subsets not containing the agent
            other_agents = [a for a in agents if a != agent]

            for r in range(len(other_agents) + 1):
                for coalition in combinations(other_agents, r):
                    coalition = set(coalition)

                    # Value with agent
                    with_agent = self._coalition_value(
                        coalition | {agent},
                        agent_contributions
                    )

                    # Value without agent
                    without_agent = self._coalition_value(
                        coalition,
                        agent_contributions
                    )

                    # Marginal contribution
                    marginal = with_agent - without_agent

                    # Weight by coalition size
                    weight = (
                        np.math.factorial(len(coalition)) *
                        np.math.factorial(self.n_agents - len(coalition) - 1) /
                        np.math.factorial(self.n_agents)
                    )

                    marginal_sum += weight * marginal
                    count += 1

            shapley_values[agent] = marginal_sum

        # Normalize to distribute global reward
        total_shapley = sum(shapley_values.values())
        if abs(total_shapley) < 1e-6:
            # Fall back to uniform if Shapley values are negligible
            return self.uniform_assignment(global_reward)

        credits = {
            agent_id: (value / total_shapley) * global_reward
            for agent_id, value in shapley_values.items()
        }

        return credits

    def _coalition_value(self, coalition: set,
                        agent_contributions: Dict[int, List[Dict]]) -> float:
        """Compute value of a coalition of agents.

        Args:
            coalition: Set of agent IDs
            agent_contributions: Contribution data

        Returns:
            Coalition value
        """
        if not coalition:
            return 0.0

        # Sum of successful tasks by coalition members
        total_successes = 0
        total_tasks = 0

        for agent_id in coalition:
            contributions = agent_contributions.get(agent_id, [])
            total_tasks += len(contributions)
            total_successes += sum(
                1 for c in contributions
                if c.get('status') == 'completed'
            )

        if total_tasks == 0:
            return 0.0

        return total_successes / total_tasks  # Success rate as value


class TemporalCreditAssigner:
    """Assign credit considering temporal dependencies.

    Handles credit assignment when agent actions have
    delayed effects on workflow outcomes.
    """

    def __init__(self, n_agents: int, gamma: float = 0.99):
        self.n_agents = n_agents
        self.gamma = gamma

    def assign_temporal_credit(self, trajectories: List[Dict],
                              final_reward: float) -> Dict[int, List[float]]:
        """Assign credit across time steps.

        Uses backward propagation of credit through the
        task dependency graph.

        Args:
            trajectories: List of (agent_id, action, immediate_reward, task_id)
            final_reward: Final workflow reward

        Returns:
            Dictionary mapping agent_id to list of step credits
        """
        agent_credits = {i: [] for i in range(self.n_agents)}

        # Backward pass through trajectory
        n_steps = len(trajectories)
        running_reward = final_reward

        for t in reversed(range(n_steps)):
            step = trajectories[t]
            agent_id = step['agent_id']
            immediate_reward = step.get('immediate_reward', 0.0)

            # Discounted credit
            credit = immediate_reward + self.gamma * running_reward / self.n_agents

            agent_credits[agent_id].insert(0, credit)

            # Update running reward
            running_reward = self.gamma * running_reward

        return agent_credits

    def compute_advantages(self, agent_credits: Dict[int, List[float]],
                          value_estimates: Dict[int, List[float]]) -> Dict[int, List[float]]:
        """Compute advantages from credits and value estimates.

        Uses GAE (Generalized Advantage Estimation).

        Args:
            agent_credits: Per-agent credits
            value_estimates: Per-agent value function estimates

        Returns:
            Per-agent advantages
        """
        lambda_gae = 0.95
        advantages = {i: [] for i in range(self.n_agents)}

        for agent_id in range(self.n_agents):
            credits = agent_credits.get(agent_id, [])
            values = value_estimates.get(agent_id, [])

            if not credits:
                continue

            # GAE computation
            gae = 0.0
            agent_advantages = []

            for t in reversed(range(len(credits))):
                if t == len(credits) - 1:
                    next_value = 0.0
                else:
                    next_value = values[t + 1] if t + 1 < len(values) else 0.0

                current_value = values[t] if t < len(values) else 0.0
                delta = credits[t] + self.gamma * next_value - current_value
                gae = delta + self.gamma * lambda_gae * gae
                agent_advantages.insert(0, gae)

            advantages[agent_id] = agent_advantages

        return advantages
