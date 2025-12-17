"""Shared Experience Buffer for Multi-Agent Learning"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional
from collections import deque
import random
import threading


class SharedExperienceBuffer:
    """Shared buffer for multi-agent experience.

    Allows agents to:
    1. Share successful execution patterns
    2. Learn from other agents' experiences
    3. Store and retrieve task-specific experiences
    """

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity

        # Global buffer (all experiences)
        self.global_buffer = deque(maxlen=capacity)

        # Per-agent buffers
        self.agent_buffers: Dict[int, deque] = {}

        # Task-type indexed buffer
        self.task_type_buffers: Dict[str, deque] = {}

        # Success/failure separated buffers
        self.success_buffer = deque(maxlen=capacity // 2)
        self.failure_buffer = deque(maxlen=capacity // 4)

        # Thread lock for concurrent access
        self._lock = threading.Lock()

    def add(self, experience: Dict[str, Any], agent_id: int = None,
            task_type: str = None, success: bool = None):
        """Add experience to appropriate buffers.

        Args:
            experience: Experience dictionary
            agent_id: ID of agent that generated this experience
            task_type: Type of task (for task-specific retrieval)
            success: Whether the task was successful
        """
        with self._lock:
            # Add to global buffer
            self.global_buffer.append(experience)

            # Add to agent-specific buffer
            if agent_id is not None:
                if agent_id not in self.agent_buffers:
                    self.agent_buffers[agent_id] = deque(maxlen=self.capacity // 10)
                self.agent_buffers[agent_id].append(experience)

            # Add to task-type buffer
            if task_type is not None:
                if task_type not in self.task_type_buffers:
                    self.task_type_buffers[task_type] = deque(maxlen=self.capacity // 10)
                self.task_type_buffers[task_type].append(experience)

            # Add to success/failure buffer
            if success is not None:
                if success:
                    self.success_buffer.append(experience)
                else:
                    self.failure_buffer.append(experience)

    def sample(self, batch_size: int, agent_id: int = None,
               task_type: str = None, success_only: bool = False) -> List[Dict]:
        """Sample experiences with optional filtering.

        Args:
            batch_size: Number of experiences to sample
            agent_id: Only sample from specific agent
            task_type: Only sample specific task type
            success_only: Only sample successful experiences

        Returns:
            List of sampled experiences
        """
        with self._lock:
            # Determine source buffer
            if success_only and self.success_buffer:
                source = list(self.success_buffer)
            elif agent_id is not None and agent_id in self.agent_buffers:
                source = list(self.agent_buffers[agent_id])
            elif task_type is not None and task_type in self.task_type_buffers:
                source = list(self.task_type_buffers[task_type])
            else:
                source = list(self.global_buffer)

            if not source:
                return []

            return random.sample(source, min(batch_size, len(source)))

    def sample_similar_tasks(self, task_embedding: torch.Tensor,
                            batch_size: int, threshold: float = 0.7) -> List[Dict]:
        """Sample experiences with similar task embeddings.

        Args:
            task_embedding: Query task embedding
            batch_size: Number of experiences to sample
            threshold: Similarity threshold

        Returns:
            Similar experiences
        """
        with self._lock:
            if not self.global_buffer:
                return []

            # Compute similarities
            similar = []
            for exp in self.global_buffer:
                if 'task_embedding' in exp:
                    sim = torch.cosine_similarity(
                        task_embedding.unsqueeze(0),
                        exp['task_embedding'].unsqueeze(0)
                    ).item()
                    if sim >= threshold:
                        similar.append((sim, exp))

            # Sort by similarity and return top-k
            similar.sort(key=lambda x: x[0], reverse=True)
            return [exp for _, exp in similar[:batch_size]]

    def get_success_rate(self, agent_id: int = None,
                        task_type: str = None) -> float:
        """Get success rate for specific agent or task type."""
        with self._lock:
            if agent_id is not None and agent_id in self.agent_buffers:
                experiences = list(self.agent_buffers[agent_id])
            elif task_type is not None and task_type in self.task_type_buffers:
                experiences = list(self.task_type_buffers[task_type])
            else:
                experiences = list(self.global_buffer)

            if not experiences:
                return 0.0

            successes = sum(1 for e in experiences if e.get('success', False))
            return successes / len(experiences)

    def get_best_strategies(self, task_type: str, top_k: int = 5) -> List[Dict]:
        """Get top-k best strategies for a task type.

        Args:
            task_type: Type of task
            top_k: Number of strategies to return

        Returns:
            List of successful strategy configurations
        """
        with self._lock:
            if task_type not in self.task_type_buffers:
                return []

            experiences = list(self.task_type_buffers[task_type])

            # Filter successful experiences
            successful = [e for e in experiences if e.get('success', False)]

            # Sort by reward
            successful.sort(
                key=lambda x: x.get('reward', 0),
                reverse=True
            )

            # Extract strategies
            strategies = []
            seen = set()
            for exp in successful:
                strategy = exp.get('strategy', {})
                strategy_key = str(strategy)
                if strategy_key not in seen:
                    strategies.append(strategy)
                    seen.add(strategy_key)
                if len(strategies) >= top_k:
                    break

            return strategies

    def clear(self):
        """Clear all buffers."""
        with self._lock:
            self.global_buffer.clear()
            self.agent_buffers.clear()
            self.task_type_buffers.clear()
            self.success_buffer.clear()
            self.failure_buffer.clear()

    def __len__(self):
        return len(self.global_buffer)

    def stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        with self._lock:
            return {
                'total_experiences': len(self.global_buffer),
                'n_agents': len(self.agent_buffers),
                'n_task_types': len(self.task_type_buffers),
                'success_count': len(self.success_buffer),
                'failure_count': len(self.failure_buffer),
                'success_rate': len(self.success_buffer) / max(
                    len(self.success_buffer) + len(self.failure_buffer), 1
                )
            }
