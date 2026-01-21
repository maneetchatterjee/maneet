"""
Quantum Detection Network (QDN) models for hyperspectral change detection.

This module contains hybrid quantum-classical neural network architectures
that combine Graph Attention Networks (GAT) with Parameterized Quantum Circuits (PQC).
"""

from .layers import GraphAttentionLayer, generate_parameters
from .qdn import (
    QDN,
    QDN_wo_QFL,
    QDN_wo_QEC,
    QDN_wo_QFL_QEC,
    GAT,
    QUEEN,
    qubit_num,
    layer_num,
)

__all__ = [
    "GraphAttentionLayer",
    "generate_parameters",
    "QDN",
    "QDN_wo_QFL",
    "QDN_wo_QEC", 
    "QDN_wo_QFL_QEC",
    "GAT",
    "QUEEN",
    "qubit_num",
    "layer_num",
]
