"""State Encoding Utilities for MARL-Flow"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional


def encode_workflow_state(workflow: Any, include_history: bool = True) -> torch.Tensor:
    """Encode workflow state into a fixed-size tensor.

    Args:
        workflow: Workflow object with tasks
        include_history: Whether to include execution history

    Returns:
        State tensor [state_dim]
    """
    state_dim = 256  # Fixed state dimension

    if workflow is None:
        return torch.zeros(state_dim)

    tasks = getattr(workflow, 'tasks', {})

    # Basic statistics
    n_tasks = len(tasks)
    n_completed = sum(1 for t in tasks.values()
                      if hasattr(t, 'status') and t.status == 'completed')
    n_pending = sum(1 for t in tasks.values()
                   if hasattr(t, 'status') and t.status == 'pending')
    n_failed = sum(1 for t in tasks.values()
                  if hasattr(t, 'status') and t.status == 'failed')

    # Dependency statistics
    total_deps = sum(len(getattr(t, 'prev', [])) for t in tasks.values())
    avg_deps = total_deps / n_tasks if n_tasks > 0 else 0

    # Progress metrics
    completion_rate = n_completed / n_tasks if n_tasks > 0 else 0
    failure_rate = n_failed / n_tasks if n_tasks > 0 else 0

    # Create feature vector
    features = [
        n_tasks / 20.0,  # Normalized task count
        n_completed / 20.0,
        n_pending / 20.0,
        n_failed / 20.0,
        completion_rate,
        failure_rate,
        avg_deps / 5.0,  # Normalized dependency count
    ]

    # Add task-level features
    if include_history:
        for i, (task_id, task) in enumerate(tasks.items()):
            if i >= 10:  # Limit to 10 tasks
                break

            status_enc = {
                'pending': 0.0,
                'completed': 1.0,
                'failed': -1.0
            }.get(getattr(task, 'status', 'pending'), 0.0)

            n_deps = len(getattr(task, 'prev', []))
            n_children = len(getattr(task, 'next', []))

            features.extend([
                status_enc,
                n_deps / 5.0,
                n_children / 5.0,
            ])

    # Pad to fixed dimension
    state = torch.zeros(state_dim)
    state[:min(len(features), state_dim)] = torch.tensor(features[:state_dim])

    return state


def encode_task_state(task: Any, context: str = "",
                     downstream_obj: str = "") -> torch.Tensor:
    """Encode task execution state.

    Args:
        task: Task object
        context: Context from upstream tasks
        downstream_obj: Downstream objectives

    Returns:
        State tensor [state_dim]
    """
    state_dim = 512  # Fixed state dimension

    if task is None:
        return torch.zeros(state_dim)

    features = []

    # Task attributes
    objective = getattr(task, 'objective', '')
    status = getattr(task, 'status', 'pending')
    agent_id = getattr(task, 'agent_id', 0)

    # Status encoding
    status_enc = {
        'pending': [1, 0, 0],
        'completed': [0, 1, 0],
        'failed': [0, 0, 1]
    }.get(status, [1, 0, 0])
    features.extend(status_enc)

    # Agent encoding (one-hot for up to 10 agents)
    agent_enc = [0.0] * 10
    if 0 <= agent_id < 10:
        agent_enc[agent_id] = 1.0
    features.extend(agent_enc)

    # Text length features (proxy for complexity)
    features.append(len(objective) / 1000.0)
    features.append(len(context) / 5000.0)
    features.append(len(downstream_obj) / 2000.0)

    # Dependencies
    n_prev = len(getattr(task, 'prev', []))
    n_next = len(getattr(task, 'next', []))
    features.append(n_prev / 5.0)
    features.append(n_next / 5.0)

    # History features
    history = getattr(task, 'history', None)
    if history is not None:
        n_attempts = len(getattr(history, 'data', []))
        features.append(n_attempts / 10.0)
    else:
        features.append(0.0)

    # Text embeddings (simplified - random for now)
    # In production, use proper text encoder
    text_emb_dim = state_dim - len(features)
    text_features = _simple_text_hash(objective, text_emb_dim)
    features.extend(text_features)

    # Create tensor
    state = torch.zeros(state_dim)
    state[:min(len(features), state_dim)] = torch.tensor(features[:state_dim])

    return state


def _simple_text_hash(text: str, dim: int) -> List[float]:
    """Create a simple hash-based embedding for text.

    This is a placeholder - use proper text encoder in production.
    """
    if not text:
        return [0.0] * dim

    # Simple hash-based features
    np.random.seed(hash(text) % (2**32))
    return list(np.random.randn(dim) * 0.1)


def get_text_embedding(text: str, model_name: str = "simple") -> torch.Tensor:
    """Get text embedding using specified model.

    Args:
        text: Text to embed
        model_name: Embedding model to use

    Returns:
        Embedding tensor [embed_dim]
    """
    embed_dim = 768  # Standard BERT dimension

    if model_name == "simple":
        # Simple hash-based embedding
        features = _simple_text_hash(text, embed_dim)
        return torch.tensor(features, dtype=torch.float32)

    elif model_name == "bert":
        # Would use actual BERT here
        # from transformers import BertTokenizer, BertModel
        raise NotImplementedError("BERT embeddings not yet implemented")

    else:
        raise ValueError(f"Unknown embedding model: {model_name}")


def compute_task_similarity(task1: Any, task2: Any) -> float:
    """Compute similarity between two tasks.

    Args:
        task1: First task
        task2: Second task

    Returns:
        Similarity score [0, 1]
    """
    obj1 = getattr(task1, 'objective', '')
    obj2 = getattr(task2, 'objective', '')

    if not obj1 or not obj2:
        return 0.0

    # Simple word overlap similarity
    words1 = set(obj1.lower().split())
    words2 = set(obj2.lower().split())

    if not words1 or not words2:
        return 0.0

    intersection = len(words1 & words2)
    union = len(words1 | words2)

    return intersection / union if union > 0 else 0.0


def batch_encode_states(states: List[Any], encode_fn: callable) -> torch.Tensor:
    """Batch encode multiple states.

    Args:
        states: List of raw states
        encode_fn: Encoding function to use

    Returns:
        Batched tensor [batch_size, state_dim]
    """
    encoded = [encode_fn(s) for s in states]
    return torch.stack(encoded)
