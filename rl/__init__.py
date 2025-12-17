"""
MARL-Flow: Multi-Agent Reinforcement Learning for Workflow Automation

This module provides RL-based enhancements to the Flow framework:
- Meta-Agent: High-level workflow structure and task scheduling
- Executor Agents: Low-level task execution strategies
- QMIX: Credit assignment for multi-agent cooperation
"""

from rl.agents.meta_agent import MetaAgent
from rl.agents.executor_agent import ExecutorAgent
from rl.rewards.reward_calculator import RewardCalculator

__all__ = ['MetaAgent', 'ExecutorAgent', 'RewardCalculator']
__version__ = '0.1.0'
