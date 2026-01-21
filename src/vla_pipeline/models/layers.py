"""
Graph Attention Layer and utility functions for QDN models.

This module provides the core building blocks for the Graph Attention Network
component of the Quantum Detection Network.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def generate_parameters(num_params: int, shape: tuple) -> list:
    """
    Generate a list of trainable parameters initialized with random values.
    
    Args:
        num_params: Number of parameter tensors to generate
        shape: Shape of each parameter tensor
        
    Returns:
        List of nn.Parameter objects
    """
    return [nn.Parameter(torch.rand(shape) * math.pi, True) for _ in range(num_params)]


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer (GAT) implementation.
    
    This layer implements the attention mechanism for graph neural networks
    as described in "Graph Attention Networks" (https://arxiv.org/abs/1710.10903).
    
    The layer computes attention coefficients between connected nodes and
    aggregates neighbor features using these attention weights.
    
    Args:
        in_features: Number of input features per node
        out_features: Number of output features per node
        dropout: Dropout probability for attention coefficients
        alpha: Negative slope for LeakyReLU activation
        concat: If True, apply ELU activation to output; otherwise return raw output
        
    Attributes:
        W: Learnable weight matrix for feature transformation
        a: Learnable attention weight vector
        leakyrelu: LeakyReLU activation function
    """
    
    def __init__(self, in_features: int, out_features: int, dropout: float, 
                 alpha: float, concat: bool = True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        # Learnable weight matrix W ∈ R^(in_features × out_features)
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        # Learnable attention weight vector a ∈ R^(2*out_features × 1)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Graph Attention Layer.
        
        Args:
            h: Node feature matrix of shape (N, in_features)
            adj: Adjacency matrix of shape (N, N)
            
        Returns:
            Updated node features of shape (N, out_features)
        """
        # Linear transformation: Wh = h · W
        Wh = torch.mm(h, self.W)  # (N, out_features)
        
        # Compute attention coefficients
        e = self._prepare_attentional_mechanism_input(Wh)

        # Mask attention coefficients for non-connected nodes
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        
        # Normalize attention coefficients using softmax
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # Aggregate neighbor features using attention weights
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh: torch.Tensor) -> torch.Tensor:
        """
        Compute attention coefficients for all node pairs.
        
        Uses additive attention: e_ij = LeakyReLU(a^T · [Wh_i || Wh_j])
        
        Args:
            Wh: Transformed node features of shape (N, out_features)
            
        Returns:
            Attention coefficient matrix of shape (N, N)
        """
        # Split attention vector into two parts for source and target nodes
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])  # (N, 1)
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])  # (N, 1)
        
        # Broadcast addition to compute attention for all pairs
        e = Wh1 + Wh2.T  # (N, N)
        
        return self.leakyrelu(e)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__} '
                f'({self.in_features} -> {self.out_features})')
