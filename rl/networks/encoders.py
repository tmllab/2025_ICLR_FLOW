"""State and Feature Encoders for MARL-Flow"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional
import numpy as np


class TaskEncoder(nn.Module):
    """Encode task descriptions into embeddings.

    Uses a transformer-based architecture to encode natural language
    task descriptions into fixed-size embeddings.
    """

    def __init__(self, vocab_size: int = 30000, embed_dim: int = 256,
                 num_heads: int = 4, num_layers: int = 2, max_len: int = 512):
        super().__init__()
        self.embed_dim = embed_dim

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_len, embed_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Task embeddings [batch_size, embed_dim]
        """
        batch_size, seq_len = input_ids.shape

        # Embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)

        # Transformer encoding
        if attention_mask is not None:
            # Convert attention mask to transformer format
            mask = attention_mask == 0
            x = self.transformer(x, src_key_padding_mask=mask)
        else:
            x = self.transformer(x)

        # Pool and project
        x = x.mean(dim=1)  # Mean pooling
        x = self.output_proj(x)

        return x


class WorkflowEncoder(nn.Module):
    """Encode workflow DAG structure into embeddings.

    Uses Graph Neural Network to encode the workflow structure
    including task nodes and dependency edges.
    """

    def __init__(self, node_dim: int = 256, edge_dim: int = 64,
                 hidden_dim: int = 256, num_layers: int = 3):
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim

        # Node feature encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Edge feature encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Message passing layers
        self.message_layers = nn.ModuleList([
            MessagePassingLayer(hidden_dim) for _ in range(num_layers)
        ])

        # Graph-level readout
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor,
                edge_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            node_features: Node feature matrix [num_nodes, node_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_features: Edge features [num_edges, edge_dim]

        Returns:
            Graph embedding [hidden_dim]
        """
        # Encode node features
        x = self.node_encoder(node_features)

        # Encode edge features
        if edge_features is not None:
            e = self.edge_encoder(edge_features)
        else:
            e = None

        # Message passing
        for layer in self.message_layers:
            x = layer(x, edge_index, e)

        # Global readout (mean pooling)
        graph_embedding = x.mean(dim=0)
        graph_embedding = self.readout(graph_embedding)

        return graph_embedding


class MessagePassingLayer(nn.Module):
    """Single message passing layer for GNN."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Node features [num_nodes, hidden_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, hidden_dim]
        """
        src, dst = edge_index

        # Compute messages
        src_features = x[src]
        dst_features = x[dst]
        messages = self.message_mlp(torch.cat([src_features, dst_features], dim=-1))

        # Aggregate messages (sum)
        aggregated = torch.zeros_like(x)
        aggregated.index_add_(0, dst, messages)

        # Update node features
        updated = self.update_mlp(torch.cat([x, aggregated], dim=-1))

        # Residual connection and layer norm
        x = self.layer_norm(x + updated)

        return x


class StateEncoder(nn.Module):
    """Unified state encoder combining task, workflow, and context information.

    This is the main encoder used by both Meta-Agent and Executor Agents.
    """

    def __init__(self, task_dim: int = 256, workflow_dim: int = 256,
                 context_dim: int = 256, output_dim: int = 512):
        super().__init__()

        # Sub-encoders
        self.task_encoder = TaskEncoder(embed_dim=task_dim)
        self.workflow_encoder = WorkflowEncoder(hidden_dim=workflow_dim)

        # Context encoder (for upstream results, etc.)
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, context_dim),
            nn.ReLU(),
            nn.Linear(context_dim, context_dim)
        )

        # Fusion layer
        total_dim = task_dim + workflow_dim + context_dim
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, task_tokens: torch.Tensor,
                workflow_nodes: torch.Tensor, workflow_edges: torch.Tensor,
                context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            task_tokens: Task description tokens [batch_size, seq_len]
            workflow_nodes: Workflow node features [num_nodes, node_dim]
            workflow_edges: Workflow edge index [2, num_edges]
            context: Context vector [batch_size, context_dim]

        Returns:
            Unified state embedding [batch_size, output_dim]
        """
        # Encode components
        task_emb = self.task_encoder(task_tokens)
        workflow_emb = self.workflow_encoder(workflow_nodes, workflow_edges)
        context_emb = self.context_encoder(context)

        # Expand workflow embedding to batch size
        batch_size = task_emb.shape[0]
        workflow_emb = workflow_emb.unsqueeze(0).expand(batch_size, -1)

        # Fuse all components
        combined = torch.cat([task_emb, workflow_emb, context_emb], dim=-1)
        state_emb = self.fusion(combined)

        return state_emb


class SimpleTextEncoder(nn.Module):
    """Simple text encoder using pretrained embeddings.

    For quick prototyping - can be replaced with BERT later.
    """

    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim

        # Simple bag-of-words style encoding
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, text_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_embedding: Pre-computed text embedding [batch_size, embed_dim]

        Returns:
            Projected embedding [batch_size, embed_dim]
        """
        return self.projection(text_embedding)
