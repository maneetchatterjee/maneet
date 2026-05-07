"""
Quantum Detection Network (QDN) - Hybrid Quantum-Classical Neural Network.

This module implements the QDN architecture for hyperspectral image change detection,
combining Graph Attention Networks (GAT) with Parameterized Quantum Circuits (PQC).

The architecture includes:
- Quantum Feature Learning (QFL): Uses quantum circuits for feature enhancement
- Quantum Enhanced Classification (QEC): Uses quantum circuits for classification
- Graph Attention Network: For spatial relationship modeling via superpixels

Reference Architecture:
    See docs/qdn_architecture.png for visual representation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False
    print("Warning: PennyLane not available. Quantum features will be disabled.")

from .layers import GraphAttentionLayer

# Quantum circuit configuration
qubit_num = 4
layer_num = 1

# Initialize quantum devices (if PennyLane is available)
if PENNYLANE_AVAILABLE:
    dev1 = qml.device("default.qubit.torch", wires=qubit_num)
    dev2 = qml.device("default.qubit.torch", wires=qubit_num)


def QUEEN(embedding: torch.Tensor, p: torch.Tensor, cp: torch.Tensor):
    """
    QUantum Enhanced Encoding Network (QUEEN) - Parameterized Quantum Circuit.
    
    This quantum circuit uses angle encoding to embed classical features into
    quantum states and applies trainable quantum gates for feature transformation.
    
    Circuit Structure (per layer):
        1. RY gates for angle encoding of input embeddings
        2. RY gates with trainable parameters
        3. IsingXX entangling gates (first pattern)
        4. RX gates with trainable parameters  
        5. IsingXX entangling gates (second pattern)
        6. RY gates with trainable parameters
        7. Multi-controlled X gates for additional entanglement
    
    Args:
        embedding: Input data embedding tensor of shape (batch, 4)
        p: Trainable rotation parameters of shape (3, num_layers, num_qubits)
        cp: Trainable entanglement parameters of shape (num_layers, num_qubits*2)
        
    Returns:
        List of expectation values for PauliZ measurements on qubits [0, 2]
    """
    measure_set = [0, 2]
    groups = [[0, 1, 2, 3]]

    embedding = embedding.unsqueeze(1) if len(embedding.shape) == 1 else embedding

    # Apply quantum layers
    for l in range(layer_num):
        for ws in groups:
            # Angle encoding with input data
            qml.RY(embedding[:, ws[0]], wires=ws[0])
            qml.RY(embedding[:, ws[1]], wires=ws[1])
            qml.RY(embedding[:, ws[2]], wires=ws[2])
            qml.RY(embedding[:, ws[3]], wires=ws[3])

            # First trainable RY rotation layer
            qml.RY(p[0, l, ws[0]], wires=ws[0])
            qml.RY(p[0, l, ws[1]], wires=ws[1])
            qml.RY(p[0, l, ws[2]], wires=ws[2])
            qml.RY(p[0, l, ws[3]], wires=ws[3])

            # IsingXX entanglement (pattern 1: adjacent pairs)
            qml.IsingXX(cp[l, ws[0]], wires=[ws[0], ws[1]])
            qml.IsingXX(cp[l, ws[1]], wires=[ws[2], ws[3]])

            # Trainable RX rotation layer
            qml.RX(p[1, l, ws[0]], wires=ws[0])
            qml.RX(p[1, l, ws[1]], wires=ws[1])
            qml.RX(p[1, l, ws[2]], wires=ws[2])
            qml.RX(p[1, l, ws[3]], wires=ws[3])

            # IsingXX entanglement (pattern 2: cross pairs)
            qml.IsingXX(cp[l, ws[2]], wires=[ws[1], ws[2]])
            qml.IsingXX(cp[l, ws[3]], wires=[ws[0], ws[3]])

            # Second trainable RY rotation layer
            qml.RY(p[2, l, ws[0]], wires=ws[0])
            qml.RY(p[2, l, ws[1]], wires=ws[1])
            qml.RY(p[2, l, ws[2]], wires=ws[2])
            qml.RY(p[2, l, ws[3]], wires=ws[3])

            # Multi-controlled X gates for additional entanglement
            qml.MultiControlledX(control_wires=[ws[0]], wires=ws[1], control_values="1")
            qml.MultiControlledX(control_wires=[ws[1]], wires=ws[2], control_values="1")
            qml.MultiControlledX(control_wires=[ws[2]], wires=ws[3], control_values="1")
            qml.MultiControlledX(control_wires=[ws[3]], wires=ws[0], control_values="1")

    # Measure expectation values
    exp_vals_z = [qml.expval(qml.PauliZ(w)) for w in measure_set]

    return exp_vals_z


def init_weights(m: nn.Module):
    """Initialize weights for linear layers using Xavier uniform initialization."""
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class GAT(nn.Module):
    """
    Multi-head Graph Attention Network module.
    
    Combines multiple attention heads and applies a final attention layer
    for output projection.
    
    Args:
        nfeat: Number of input features
        nhid: Number of hidden features per attention head
        A: Adjacency matrix for the graph
        nout: Number of output features
        dropout: Dropout probability
        alpha: Negative slope for LeakyReLU
        nheads: Number of attention heads
    """
    
    def __init__(self, nfeat: int, nhid: int, A: torch.Tensor, nout: int,
                 dropout: float, alpha: float, nheads: int):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.A = A
        
        # Multi-head attention layers
        self.attentions = [
            GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True)
            for _ in range(nheads)
        ]
        for i, attention in enumerate(self.attentions):
            self.add_module(f'attention_{i}', attention)

        # Output attention layer
        self.out_att = GraphAttentionLayer(
            nhid * nheads, nout, dropout=dropout, alpha=alpha, concat=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the GAT network.
        
        Args:
            x: Node feature matrix of shape (N, nfeat)
            
        Returns:
            Updated node features of shape (N, nout)
        """
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, self.A) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, self.A))
        return x


class QDN(nn.Module):
    """
    Quantum Detection Network - Full model with QFL and QEC.
    
    This is the complete hybrid quantum-classical architecture combining:
    - Graph Attention Network for spatial feature extraction
    - Quantum Feature Learning (QFL) for quantum-enhanced feature encoding
    - Quantum Enhanced Classification (QEC) for quantum-assisted classification
    - Attention-based fusion of classical and quantum predictions
    
    Args:
        channel: Number of input channels (spectral bands)
        class_num: Number of output classes (typically 2 for change/no-change)
        Q: Superpixel assignment matrix (pixels × superpixels)
        A: Superpixel adjacency matrix (superpixels × superpixels)
        model: Model variant identifier (default: 'normal')
        
    Architecture Overview:
        Input X1, X2 (H×W×C) → Dim Reduction → Difference → 
        [Graph Branch: Superpixel → GAT → Project] + 
        [QFL Branch: Conv → QUEEN → Conv] →
        Feature Fusion → Conv Layers →
        [Classical: Linear → Softmax] + [QEC: Conv → QUEEN] →
        Attention Fusion → Final Prediction
    """
    
    def __init__(self, channel: int, class_num: int, Q: torch.Tensor, 
                 A: torch.Tensor, model: str = 'normal'):
        super(QDN, self).__init__()
        
        if not PENNYLANE_AVAILABLE:
            raise ImportError("PennyLane is required for QDN model")
        
        self.Q = Q
        self.A = A
        self.model = model
        self.norm_col_Q = Q / (torch.sum(Q, 0, keepdim=True))
        self.channel = channel

        # Graph Attention Network
        self.GATNet1 = GAT(
            nfeat=64, nhid=128, A=A, nout=64, 
            dropout=0.4, nheads=2, alpha=0.2
        )

        # Feature extraction layers
        self.extract = nn.Sequential()
        self.extract.add_module(
            'feature_extractor',
            torch.nn.Conv2d(64 * 3 + 1, 64, kernel_size=(3, 3), padding=1)
        )
        self.extract.apply(init_weights)

        self.conv = nn.Sequential()
        self.conv.add_module(
            'Conv',
            torch.nn.Conv2d(64, 32, kernel_size=(3, 3), padding=1)
        )
        self.conv.apply(init_weights)

        # Dimension reduction convolutions
        self.dim_reduc_conv1 = nn.Sequential()
        self.dim_reduc_conv1.add_module(
            'Conv',
            torch.nn.Conv2d(channel, 64, kernel_size=(1, 1), padding=0)
        )
        self.dim_reduc_conv1.apply(init_weights)

        self.dim_reduc_conv2 = nn.Sequential()
        self.dim_reduc_conv2.add_module(
            'Conv',
            torch.nn.Conv2d(channel, 64, kernel_size=(1, 1), padding=0)
        )
        self.dim_reduc_conv2.apply(init_weights)

        # Linear and activation layers
        self.linear1 = nn.Linear(128, 64)
        self.linear1.apply(init_weights)
        self.act1 = nn.PReLU()
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(32)

        self.first_in_channel = 64
        self.in_channel = 64

        # QFL (Quantum Feature Learning) parameters
        self.p = nn.Parameter(
            torch.rand((3, layer_num, qubit_num)) * math.pi, True
        )
        self.cp = nn.Parameter(
            torch.zeros((layer_num, qubit_num * 2)) * math.pi, True
        )
        self.qnn = qml.QNode(QUEEN, dev1, interface='torch', diff_method='backprop')

        # QEC (Quantum Enhanced Classification) parameters
        self.p1 = nn.Parameter(torch.rand((3, 1, 4)) * math.pi, True)
        self.cp1 = nn.Parameter(torch.zeros((1, 4)) * math.pi, True)
        self.qnn1 = qml.QNode(QUEEN, dev2, interface='torch', diff_method='backprop')

        # QNN dimension reduction
        self.QNN_red = nn.Sequential()
        self.QNN_red.add_module(
            'Conv',
            torch.nn.Conv2d(32, 4, kernel_size=(1, 1), padding=0)
        )
        self.QNN_red.apply(init_weights)

        # Attention weights for fusion
        self.Watt = nn.Parameter(torch.empty(size=(1, 4)))
        nn.init.xavier_uniform_(self.Watt.data, gain=1.414)

        # Classification layers
        self.Softmax_linear = nn.Sequential(nn.Linear(32, 2))
        self.Softmax_linear.apply(init_weights)
        self.fuse_classify = nn.Sequential(nn.Linear(4, 2))
        self.fuse_classify.apply(init_weights)

        # QFL dimension processing
        self.DR3 = nn.Sequential()
        self.DR3.add_module(
            'Conv',
            torch.nn.Conv2d(64, 8, kernel_size=(1, 1), padding=0)
        )
        self.DR3.apply(init_weights)
        
        self.spec_up = nn.Sequential()
        self.spec_up.add_module(
            'Conv',
            torch.nn.Conv2d(4, 64, kernel_size=(1, 1), padding=0)
        )
        self.spec_up.apply(init_weights)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, 
                spectral_map: torch.Tensor):
        """
        Forward pass of the QDN model.
        
        Args:
            x1: First hyperspectral image (H, W, C)
            x2: Second hyperspectral image (H, W, C)
            spectral_map: Spectral attention map (H*W, 1)
            
        Returns:
            Tuple of (final_prediction, classical_prediction, quantum_prediction)
            Each prediction has shape (H*W, 2)
        """
        (h, w, c) = x1.shape

        # Dimension reduction for both inputs
        x1 = self.dim_reduc_conv1(torch.unsqueeze(x1.permute([2, 0, 1]), 0))
        x1 = torch.squeeze(x1, 0).permute([1, 2, 0]).reshape([h * w, -1])
        x1 = self.act1(x1)

        x2 = self.dim_reduc_conv2(torch.unsqueeze(x2.permute([2, 0, 1]), 0))
        x2 = torch.squeeze(x2, 0).permute([1, 2, 0]).reshape([h * w, -1])
        x2 = self.act1(x2)

        # Compute difference features
        x_diff = x1 - x2

        # === Graph Attention Branch ===
        x_diff_flatten = x_diff.reshape([h * w, -1])
        superpixels_flatten1 = torch.mm(self.norm_col_Q.t(), x_diff_flatten)

        H1 = superpixels_flatten1
        H1 = self.GATNet1(H1)

        feature_map1 = torch.matmul(self.Q, H1)

        # === Quantum Feature Learning (QFL) Branch ===
        x_diff_1 = torch.unsqueeze(
            x_diff.reshape([h, w, 64]).permute([2, 0, 1]), 0
        )

        Y = self.DR3(x_diff_1)
        Y = torch.squeeze(Y, 0).permute([1, 2, 0]).reshape([h * w, -1])
        Y = self.act1(Y)
        # Reshape to feed pairs of 4-dimensional vectors to quantum circuit
        # DR3 outputs 8 channels, split into 2 groups of 4 for quantum processing
        Y = Y.reshape([int(h * w * 2), 4])
        
        xq = self.qnn(Y, self.p, self.cp)
        xq = torch.stack(xq)
        xq = torch.transpose(xq, 0, 1).reshape([h * w, 4])
        xq = xq.float()
        
        x_diff_1 = torch.unsqueeze(
            xq.reshape([h, w, -1]).permute([2, 0, 1]), 0
        )
        x_diff_1 = self.spec_up(x_diff_1)
        x_diff_re_flatten = torch.squeeze(x_diff_1, 0).permute([1, 2, 0]).reshape([h * w, -1])

        # === Feature Fusion ===
        feature_map1 = feature_map1 + x_diff_re_flatten
        feature_map1 = spectral_map * feature_map1

        x = torch.cat((feature_map1, x1, x2, spectral_map), -1)

        # === Feature Extraction ===
        x1 = self.extract(
            torch.unsqueeze(x.reshape([h, w, 64 * 3 + 1]).permute([2, 0, 1]), 0)
        )
        x1 = torch.squeeze(x1, 0).permute([1, 2, 0]).reshape([h * w, -1])
        x2 = self.bn1(self.act1(x1))
        
        x3 = self.conv(
            torch.unsqueeze(x2.reshape([h, w, 64]).permute([2, 0, 1]), 0)
        )
        x3 = torch.squeeze(x3, 0).permute([1, 2, 0]).reshape([h * w, -1])

        x4 = self.bn2(self.act1(x3))
        
        # === Classical Classification Branch ===
        x4 = self.Softmax_linear(x4)
        x4_mid = F.softmax(x4, -1)

        # === Quantum Enhanced Classification (QEC) Branch ===
        Y = self.QNN_red(
            torch.unsqueeze(x3.reshape([h, w, 32]).permute([2, 0, 1]), 0)
        )
        Y = torch.squeeze(Y, 0).permute([1, 2, 0]).reshape([h * w, -1])
        Y = self.act1(Y)
        Y = Y.reshape([int(h * w), 4])
        
        xq = self.qnn1(Y, self.p1, self.cp1)
        xq = torch.stack(xq)
        xq = torch.transpose(xq, 0, 1).reshape([h * w, 2])
        xq = xq.float()

        xq_mid = F.softmax(xq, -1)

        # === Attention-based Fusion ===
        Y = torch.cat((xq, x4), -1)

        Watt = F.softmax(self.Watt, dim=-1)
        Y = Y * Watt
        Y = self.fuse_classify(Y)
        Y = F.softmax(Y, -1)

        return Y, x4_mid, xq_mid


class QDN_wo_QFL_QEC(nn.Module):
    """
    QDN variant without Quantum Feature Learning and Quantum Enhanced Classification.
    
    This is a purely classical baseline that uses only:
    - Graph Attention Network for spatial features
    - Classical convolutional layers for classification
    
    Useful for ablation studies comparing quantum vs classical approaches.
    """
    
    def __init__(self, channel: int, class_num: int, Q: torch.Tensor, 
                 A: torch.Tensor, model: str = 'normal'):
        super(QDN_wo_QFL_QEC, self).__init__()
        self.Q = Q
        self.A = A
        self.model = model
        self.norm_col_Q = Q / (torch.sum(Q, 0, keepdim=True))
        self.channel = channel

        self.GATNet1 = GAT(
            nfeat=64, nhid=128, A=A, nout=64,
            dropout=0.4, nheads=2, alpha=0.2
        )

        self.extract = nn.Sequential()
        self.extract.add_module(
            'feature_extractor',
            torch.nn.Conv2d(64 * 3 + 1, 64, kernel_size=(3, 3), padding=1)
        )
        self.extract.apply(init_weights)

        self.conv = nn.Sequential()
        self.conv.add_module(
            'Conv',
            torch.nn.Conv2d(64, 32, kernel_size=(3, 3), padding=1)
        )
        self.conv.apply(init_weights)

        self.dim_reduc_conv1 = nn.Sequential()
        self.dim_reduc_conv1.add_module(
            'Conv',
            torch.nn.Conv2d(channel, 64, kernel_size=(1, 1), padding=0)
        )
        self.dim_reduc_conv1.apply(init_weights)

        self.dim_reduc_conv2 = nn.Sequential()
        self.dim_reduc_conv2.add_module(
            'Conv',
            torch.nn.Conv2d(channel, 64, kernel_size=(1, 1), padding=0)
        )
        self.dim_reduc_conv2.apply(init_weights)

        self.linear1 = nn.Linear(128, 64)
        self.linear1.apply(init_weights)
        self.act1 = nn.PReLU()
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(32)

        self.first_in_channel = 64
        self.in_channel = 64

        self.Softmax_linear = nn.Sequential(nn.Linear(32, 2))
        self.Softmax_linear.apply(init_weights)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, 
                spectral_map: torch.Tensor) -> torch.Tensor:
        """Forward pass returning only classical prediction."""
        (h, w, c) = x1.shape

        x1 = self.dim_reduc_conv1(torch.unsqueeze(x1.permute([2, 0, 1]), 0))
        x1 = torch.squeeze(x1, 0).permute([1, 2, 0]).reshape([h * w, -1])
        x1 = self.act1(x1)

        x2 = self.dim_reduc_conv2(torch.unsqueeze(x2.permute([2, 0, 1]), 0))
        x2 = torch.squeeze(x2, 0).permute([1, 2, 0]).reshape([h * w, -1])
        x2 = self.act1(x2)

        x_diff = x1 - x2

        x_diff_flatten = x_diff.reshape([h * w, -1])
        superpixels_flatten1 = torch.mm(self.norm_col_Q.t(), x_diff_flatten)

        H1 = superpixels_flatten1
        H1 = self.GATNet1(H1)

        feature_map1 = torch.matmul(self.Q, H1)
        feature_map1 = spectral_map * feature_map1

        x = torch.cat((feature_map1, x1, x2, spectral_map), -1)

        x1 = self.extract(
            torch.unsqueeze(x.reshape([h, w, 64 * 3 + 1]).permute([2, 0, 1]), 0)
        )
        x1 = torch.squeeze(x1, 0).permute([1, 2, 0]).reshape([h * w, -1])
        x2 = self.bn1(self.act1(x1))
        
        x3 = self.conv(
            torch.unsqueeze(x2.reshape([h, w, 64]).permute([2, 0, 1]), 0)
        )
        x3 = torch.squeeze(x3, 0).permute([1, 2, 0]).reshape([h * w, -1])

        x4 = self.bn2(self.act1(x3))
        x4 = self.Softmax_linear(x4)
        x4_mid = F.softmax(x4, -1)

        return x4_mid


class QDN_wo_QFL(nn.Module):
    """
    QDN variant without Quantum Feature Learning (QFL).
    
    Uses only QEC (Quantum Enhanced Classification) for quantum processing.
    The feature extraction remains classical (GAT + Conv).
    """
    
    def __init__(self, channel: int, class_num: int, Q: torch.Tensor, 
                 A: torch.Tensor, model: str = 'normal'):
        super(QDN_wo_QFL, self).__init__()
        
        if not PENNYLANE_AVAILABLE:
            raise ImportError("PennyLane is required for QDN_wo_QFL model")
        
        self.Q = Q
        self.A = A
        self.model = model
        self.norm_col_Q = Q / (torch.sum(Q, 0, keepdim=True))
        self.channel = channel

        self.GATNet1 = GAT(
            nfeat=64, nhid=128, A=A, nout=64,
            dropout=0.4, nheads=2, alpha=0.2
        )

        self.extract = nn.Sequential()
        self.extract.add_module(
            'feature_extractor',
            torch.nn.Conv2d(64 * 3 + 1, 64, kernel_size=(3, 3), padding=1)
        )
        self.extract.apply(init_weights)

        self.conv = nn.Sequential()
        self.conv.add_module(
            'Conv',
            torch.nn.Conv2d(64, 32, kernel_size=(3, 3), padding=1)
        )
        self.conv.apply(init_weights)

        self.dim_reduc_conv1 = nn.Sequential()
        self.dim_reduc_conv1.add_module(
            'Conv',
            torch.nn.Conv2d(channel, 64, kernel_size=(1, 1), padding=0)
        )
        self.dim_reduc_conv1.apply(init_weights)

        self.dim_reduc_conv2 = nn.Sequential()
        self.dim_reduc_conv2.add_module(
            'Conv',
            torch.nn.Conv2d(channel, 64, kernel_size=(1, 1), padding=0)
        )
        self.dim_reduc_conv2.apply(init_weights)

        self.linear1 = nn.Linear(128, 64)
        self.linear1.apply(init_weights)
        self.act1 = nn.PReLU()
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(32)

        self.first_in_channel = 64
        self.in_channel = 64

        # QEC parameters
        self.p1 = nn.Parameter(torch.rand((3, 1, 4)) * math.pi, True)
        self.cp1 = nn.Parameter(torch.zeros((1, 4)) * math.pi, True)
        self.qnn1 = qml.QNode(QUEEN, dev2, interface='torch', diff_method='backprop')

        self.QNN_red = nn.Sequential()
        self.QNN_red.add_module(
            'Conv',
            torch.nn.Conv2d(32, 4, kernel_size=(1, 1), padding=0)
        )
        self.QNN_red.apply(init_weights)

        self.Watt = nn.Parameter(torch.empty(size=(1, 4)))
        nn.init.xavier_uniform_(self.Watt.data, gain=1.414)

        self.Softmax_linear = nn.Sequential(nn.Linear(32, 2))
        self.Softmax_linear.apply(init_weights)
        self.fuse_classify = nn.Sequential(nn.Linear(4, 2))
        self.fuse_classify.apply(init_weights)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, 
                spectral_map: torch.Tensor):
        """Forward pass with QEC but without QFL."""
        (h, w, c) = x1.shape

        x1 = self.dim_reduc_conv1(torch.unsqueeze(x1.permute([2, 0, 1]), 0))
        x1 = torch.squeeze(x1, 0).permute([1, 2, 0]).reshape([h * w, -1])
        x1 = self.act1(x1)

        x2 = self.dim_reduc_conv2(torch.unsqueeze(x2.permute([2, 0, 1]), 0))
        x2 = torch.squeeze(x2, 0).permute([1, 2, 0]).reshape([h * w, -1])
        x2 = self.act1(x2)

        x_diff = x1 - x2

        x_diff_flatten = x_diff.reshape([h * w, -1])
        superpixels_flatten1 = torch.mm(self.norm_col_Q.t(), x_diff_flatten)

        H1 = superpixels_flatten1
        H1 = self.GATNet1(H1)

        feature_map1 = torch.matmul(self.Q, H1)
        feature_map1 = spectral_map * feature_map1

        x = torch.cat((feature_map1, x1, x2, spectral_map), -1)

        x1 = self.extract(
            torch.unsqueeze(x.reshape([h, w, 64 * 3 + 1]).permute([2, 0, 1]), 0)
        )
        x1 = torch.squeeze(x1, 0).permute([1, 2, 0]).reshape([h * w, -1])
        x2 = self.bn1(self.act1(x1))
        
        x3 = self.conv(
            torch.unsqueeze(x2.reshape([h, w, 64]).permute([2, 0, 1]), 0)
        )
        x3 = torch.squeeze(x3, 0).permute([1, 2, 0]).reshape([h * w, -1])

        x4 = self.bn2(self.act1(x3))
        x4 = self.Softmax_linear(x4)
        x4_mid = F.softmax(x4, -1)

        # QEC Branch
        Y = self.QNN_red(
            torch.unsqueeze(x3.reshape([h, w, 32]).permute([2, 0, 1]), 0)
        )
        Y = torch.squeeze(Y, 0).permute([1, 2, 0]).reshape([h * w, -1])
        Y = self.act1(Y)
        Y = Y.reshape([int(h * w), 4])
        
        xq = self.qnn1(Y, self.p1, self.cp1)
        xq = torch.stack(xq)
        xq = torch.transpose(xq, 0, 1).reshape([h * w, 2])
        xq = xq.float()

        xq_mid = F.softmax(xq, -1)

        Y = torch.cat((xq, x4), -1)

        Watt = F.softmax(self.Watt, dim=-1)
        Y = Y * Watt
        Y = self.fuse_classify(Y)
        Y = F.softmax(Y, -1)

        return Y, x4_mid, xq_mid


class QDN_wo_QEC(nn.Module):
    """
    QDN variant without Quantum Enhanced Classification (QEC).
    
    Uses QFL (Quantum Feature Learning) for quantum-enhanced features
    but relies on classical classification.
    """
    
    def __init__(self, channel: int, class_num: int, Q: torch.Tensor, 
                 A: torch.Tensor, model: str = 'normal'):
        super(QDN_wo_QEC, self).__init__()
        
        if not PENNYLANE_AVAILABLE:
            raise ImportError("PennyLane is required for QDN_wo_QEC model")
        
        self.Q = Q
        self.A = A
        self.model = model
        self.norm_col_Q = Q / (torch.sum(Q, 0, keepdim=True))
        self.channel = channel

        self.GATNet1 = GAT(
            nfeat=64, nhid=128, A=A, nout=64,
            dropout=0.4, nheads=2, alpha=0.2
        )

        self.extract = nn.Sequential()
        self.extract.add_module(
            'feature_extractor',
            torch.nn.Conv2d(64 * 3 + 1, 64, kernel_size=(3, 3), padding=1)
        )
        self.extract.apply(init_weights)

        self.conv = nn.Sequential()
        self.conv.add_module(
            'Conv',
            torch.nn.Conv2d(64, 32, kernel_size=(3, 3), padding=1)
        )
        self.conv.apply(init_weights)

        self.dim_reduc_conv1 = nn.Sequential()
        self.dim_reduc_conv1.add_module(
            'Conv',
            torch.nn.Conv2d(channel, 64, kernel_size=(1, 1), padding=0)
        )
        self.dim_reduc_conv1.apply(init_weights)

        self.dim_reduc_conv2 = nn.Sequential()
        self.dim_reduc_conv2.add_module(
            'Conv',
            torch.nn.Conv2d(channel, 64, kernel_size=(1, 1), padding=0)
        )
        self.dim_reduc_conv2.apply(init_weights)

        self.linear1 = nn.Linear(128, 64)
        self.linear1.apply(init_weights)
        self.act1 = nn.PReLU()
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(32)

        self.first_in_channel = 64
        self.in_channel = 64

        # QFL parameters
        self.p = nn.Parameter(
            torch.rand((3, layer_num, qubit_num)) * math.pi, True
        )
        self.cp = nn.Parameter(
            torch.zeros((layer_num, qubit_num * 2)) * math.pi, True
        )
        self.qnn = qml.QNode(QUEEN, dev1, interface='torch', diff_method='backprop')

        self.Softmax_linear = nn.Sequential(nn.Linear(32, 2))
        self.Softmax_linear.apply(init_weights)

        self.DR3 = nn.Sequential()
        self.DR3.add_module(
            'Conv',
            torch.nn.Conv2d(64, 8, kernel_size=(1, 1), padding=0)
        )
        self.DR3.apply(init_weights)
        
        self.spec_up = nn.Sequential()
        self.spec_up.add_module(
            'Conv',
            torch.nn.Conv2d(4, 64, kernel_size=(1, 1), padding=0)
        )
        self.spec_up.apply(init_weights)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, 
                spectral_map: torch.Tensor) -> torch.Tensor:
        """Forward pass with QFL but without QEC."""
        (h, w, c) = x1.shape

        x1 = self.dim_reduc_conv1(torch.unsqueeze(x1.permute([2, 0, 1]), 0))
        x1 = torch.squeeze(x1, 0).permute([1, 2, 0]).reshape([h * w, -1])
        x1 = self.act1(x1)

        x2 = self.dim_reduc_conv2(torch.unsqueeze(x2.permute([2, 0, 1]), 0))
        x2 = torch.squeeze(x2, 0).permute([1, 2, 0]).reshape([h * w, -1])
        x2 = self.act1(x2)

        x_diff = x1 - x2

        x_diff_flatten = x_diff.reshape([h * w, -1])
        superpixels_flatten1 = torch.mm(self.norm_col_Q.t(), x_diff_flatten)

        H1 = superpixels_flatten1
        H1 = self.GATNet1(H1)

        feature_map1 = torch.matmul(self.Q, H1)

        # QFL Branch
        x_diff_1 = torch.unsqueeze(
            x_diff.reshape([h, w, 64]).permute([2, 0, 1]), 0
        )

        Y = self.DR3(x_diff_1)
        Y = torch.squeeze(Y, 0).permute([1, 2, 0]).reshape([h * w, -1])
        Y = self.act1(Y)
        # Reshape to feed pairs of 4-dimensional vectors to quantum circuit
        # DR3 outputs 8 channels, split into 2 groups of 4 for quantum processing
        Y = Y.reshape([int(h * w * 2), 4])
        
        xq = self.qnn(Y, self.p, self.cp)
        xq = torch.stack(xq)
        xq = torch.transpose(xq, 0, 1).reshape([h * w, 4])
        xq = xq.float()
        
        x_diff_1 = torch.unsqueeze(
            xq.reshape([h, w, -1]).permute([2, 0, 1]), 0
        )
        x_diff_1 = self.spec_up(x_diff_1)
        x_diff_re_flatten = torch.squeeze(x_diff_1, 0).permute([1, 2, 0]).reshape([h * w, -1])

        feature_map1 = feature_map1 + x_diff_re_flatten
        feature_map1 = spectral_map * feature_map1

        x = torch.cat((feature_map1, x1, x2, spectral_map), -1)

        x1 = self.extract(
            torch.unsqueeze(x.reshape([h, w, 64 * 3 + 1]).permute([2, 0, 1]), 0)
        )
        x1 = torch.squeeze(x1, 0).permute([1, 2, 0]).reshape([h * w, -1])
        x2 = self.bn1(self.act1(x1))
        
        x3 = self.conv(
            torch.unsqueeze(x2.reshape([h, w, 64]).permute([2, 0, 1]), 0)
        )
        x3 = torch.squeeze(x3, 0).permute([1, 2, 0]).reshape([h * w, -1])

        x4 = self.bn2(self.act1(x3))
        x4 = self.Softmax_linear(x4)
        x4_mid = F.softmax(x4, -1)

        return x4_mid
