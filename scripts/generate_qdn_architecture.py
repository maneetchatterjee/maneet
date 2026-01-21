#!/usr/bin/env python3
"""
Generate QDN (Quantum Detection Network) architecture diagram.

This script creates a visualization of the hybrid quantum-classical neural network
architecture for change detection in hyperspectral images.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
import numpy as np

def draw_rounded_box(ax, x, y, width, height, text, color='lightblue', 
                     fontsize=9, text_color='black', alpha=0.8):
    """Draw a rounded rectangle box with text."""
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                         boxstyle="round,pad=0.02,rounding_size=0.1",
                         facecolor=color, edgecolor='black', linewidth=1.5,
                         alpha=alpha)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            fontweight='bold', color=text_color, wrap=True)

def draw_arrow(ax, start, end, color='black', style='->', connectionstyle="arc3,rad=0"):
    """Draw an arrow between two points."""
    arrow = FancyArrowPatch(start, end, arrowstyle=style, color=color,
                           connectionstyle=connectionstyle, linewidth=1.5,
                           mutation_scale=15)
    ax.add_patch(arrow)

def draw_quantum_circuit_box(ax, x, y, width, height):
    """Draw a quantum circuit representation."""
    # Main box
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                         boxstyle="round,pad=0.02,rounding_size=0.05",
                         facecolor='#E8E0F0', edgecolor='purple', linewidth=2,
                         alpha=0.9)
    ax.add_patch(box)
    
    # Draw qubit lines
    qubit_y_positions = [y + height/2 - height*(i+1)/5 for i in range(4)]
    for qy in qubit_y_positions:
        ax.plot([x - width/2 + 0.05, x + width/2 - 0.05], [qy, qy], 
                'k-', linewidth=0.8, alpha=0.6)
    
    # Draw gate symbols
    gate_x_positions = [x - width/4, x, x + width/4]
    for gx in gate_x_positions:
        for qy in qubit_y_positions:
            circle = Circle((gx, qy), 0.02, facecolor='purple', 
                           edgecolor='darkblue', alpha=0.7)
            ax.add_patch(circle)

def create_qdn_architecture():
    """Create the complete QDN architecture diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 9.5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Title
    ax.text(5, 9.2, 'QDN: Quantum Detection Network Architecture', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    ax.text(5, 8.8, 'Hybrid Quantum-Classical Network for Hyperspectral Change Detection', 
            ha='center', va='center', fontsize=11, style='italic', color='gray')
    
    # === INPUT SECTION ===
    # Input Images
    draw_rounded_box(ax, 1, 7.5, 1.2, 0.6, 'Input X₁\n(H×W×C)', '#FFE4B5', 8)
    draw_rounded_box(ax, 3, 7.5, 1.2, 0.6, 'Input X₂\n(H×W×C)', '#FFE4B5', 8)
    draw_rounded_box(ax, 5, 7.5, 1.4, 0.6, 'Spectral Map\n(H×W×1)', '#98FB98', 8)
    
    # === DIMENSION REDUCTION ===
    draw_rounded_box(ax, 1, 6.3, 1.4, 0.5, 'Dim Reduc Conv1\n(C→64)', '#87CEEB', 8)
    draw_rounded_box(ax, 3, 6.3, 1.4, 0.5, 'Dim Reduc Conv2\n(C→64)', '#87CEEB', 8)
    
    draw_arrow(ax, (1, 7.2), (1, 6.55))
    draw_arrow(ax, (3, 7.2), (3, 6.55))
    
    # PReLU Activations
    draw_rounded_box(ax, 1, 5.7, 0.8, 0.35, 'PReLU', '#FFA07A', 7)
    draw_rounded_box(ax, 3, 5.7, 0.8, 0.35, 'PReLU', '#FFA07A', 7)
    
    draw_arrow(ax, (1, 6.05), (1, 5.9))
    draw_arrow(ax, (3, 6.05), (3, 5.9))
    
    # === DIFFERENCE COMPUTATION ===
    draw_rounded_box(ax, 2, 5.0, 1.2, 0.5, 'X_diff = X₁ - X₂', '#DDA0DD', 8)
    draw_arrow(ax, (1, 5.5), (1.7, 5.25), connectionstyle="arc3,rad=-0.2")
    draw_arrow(ax, (3, 5.5), (2.3, 5.25), connectionstyle="arc3,rad=0.2")
    
    # === GRAPH ATTENTION BRANCH ===
    ax.text(1, 4.4, 'Graph Branch', ha='center', fontsize=9, fontweight='bold', color='#2E8B57')
    
    draw_rounded_box(ax, 1, 3.9, 1.6, 0.5, 'Superpixel Pool\n(Q^T · X_diff)', '#90EE90', 7)
    draw_arrow(ax, (1.5, 4.75), (1, 4.15), connectionstyle="arc3,rad=0.2")
    
    # GAT Network
    gat_box = FancyBboxPatch((0.15, 2.9), 1.7, 0.8,
                             boxstyle="round,pad=0.02,rounding_size=0.1",
                             facecolor='#98FB98', edgecolor='#2E8B57', linewidth=2)
    ax.add_patch(gat_box)
    ax.text(1, 3.45, 'GAT Network', ha='center', fontsize=9, fontweight='bold')
    ax.text(1, 3.15, '(64→128→64)\n2 Heads', ha='center', fontsize=7)
    
    draw_arrow(ax, (1, 3.65), (1, 3.7))
    
    draw_rounded_box(ax, 1, 2.4, 1.4, 0.4, 'Feature Proj\n(Q · H)', '#90EE90', 7)
    draw_arrow(ax, (1, 2.9), (1, 2.6))
    
    # === QUANTUM FEATURE LEARNING (QFL) BRANCH ===
    ax.text(4.5, 4.4, 'Quantum Feature Learning (QFL)', ha='center', fontsize=9, 
            fontweight='bold', color='#8B008B')
    
    draw_rounded_box(ax, 4.5, 3.9, 1.4, 0.5, 'DR3 Conv\n(64→8)', '#E6E6FA', 8)
    draw_arrow(ax, (2.5, 4.75), (4.1, 4.15), connectionstyle="arc3,rad=-0.1")
    
    # Quantum Circuit 1 (QNN)
    qnn1_x, qnn1_y = 4.5, 3.0
    draw_quantum_circuit_box(ax, qnn1_x, qnn1_y, 1.6, 0.8)
    ax.text(qnn1_x, qnn1_y + 0.15, 'QUEEN Circuit', ha='center', fontsize=8, 
            fontweight='bold', color='purple')
    ax.text(qnn1_x, qnn1_y - 0.15, '4 Qubits × 1 Layer', ha='center', fontsize=7)
    draw_arrow(ax, (4.5, 3.65), (4.5, 3.4))
    
    draw_rounded_box(ax, 4.5, 2.2, 1.4, 0.5, 'Spec Up Conv\n(4→64)', '#E6E6FA', 8)
    draw_arrow(ax, (4.5, 2.6), (4.5, 2.45))
    
    # === FUSION OF GAT + QFL ===
    draw_rounded_box(ax, 2.75, 1.5, 2.0, 0.5, 'Feature Fusion (+)\n+ Spectral Weighting', '#FFB6C1', 8)
    draw_arrow(ax, (1, 2.2), (2.2, 1.75), connectionstyle="arc3,rad=0.1")
    draw_arrow(ax, (4.5, 1.95), (3.3, 1.75), connectionstyle="arc3,rad=-0.1")
    draw_arrow(ax, (5, 7.2), (5, 6.5), connectionstyle="arc3,rad=0")
    ax.plot([5, 5], [6.5, 1.75], 'k--', linewidth=0.8, alpha=0.5)
    draw_arrow(ax, (5, 1.75), (3.75, 1.65), connectionstyle="arc3,rad=0.3")
    
    # === FEATURE CONCATENATION ===
    draw_rounded_box(ax, 7, 5.5, 2.0, 0.6, 'Concatenate\n(Feature, X₁, X₂, Spec)', '#FFFACD', 8)
    draw_arrow(ax, (2.75, 1.25), (6, 1.0), connectionstyle="arc3,rad=-0.3")
    ax.plot([6, 6.5], [1.0, 5.2], 'k-', linewidth=1.2)
    draw_arrow(ax, (6.5, 5.2), (6.0, 5.35), connectionstyle="arc3,rad=0.1")
    
    # === FEATURE EXTRACTION ===
    draw_rounded_box(ax, 7, 4.7, 1.6, 0.5, 'Extract Conv\n(193→64, 3×3)', '#87CEEB', 8)
    draw_arrow(ax, (7, 5.2), (7, 4.95))
    
    draw_rounded_box(ax, 7, 4.0, 1.0, 0.35, 'BN + PReLU', '#FFA07A', 7)
    draw_arrow(ax, (7, 4.45), (7, 4.2))
    
    draw_rounded_box(ax, 7, 3.3, 1.4, 0.5, 'Conv Layer\n(64→32, 3×3)', '#87CEEB', 8)
    draw_arrow(ax, (7, 3.8), (7, 3.55))
    
    draw_rounded_box(ax, 7, 2.6, 1.0, 0.35, 'BN + PReLU', '#FFA07A', 7)
    draw_arrow(ax, (7, 3.05), (7, 2.8))
    
    # === CLASSICAL BRANCH ===
    ax.text(6, 2.0, 'Classical\nClassifier', ha='center', fontsize=8, fontweight='bold', color='#4169E1')
    draw_rounded_box(ax, 6, 1.4, 1.2, 0.5, 'Linear\n(32→2)', '#ADD8E6', 8)
    draw_arrow(ax, (6.5, 2.45), (6, 1.65), connectionstyle="arc3,rad=0.2")
    
    draw_rounded_box(ax, 6, 0.7, 1.0, 0.35, 'Softmax', '#B0E0E6', 7)
    draw_arrow(ax, (6, 1.15), (6, 0.9))
    
    # === QEC BRANCH (Quantum Enhanced Classification) ===
    ax.text(8.5, 2.0, 'Quantum Enhanced\nClassification (QEC)', ha='center', fontsize=8, 
            fontweight='bold', color='#8B008B')
    
    draw_rounded_box(ax, 8.5, 1.4, 1.2, 0.5, 'QNN Red Conv\n(32→4)', '#E6E6FA', 8)
    draw_arrow(ax, (7.5, 2.45), (8.5, 1.65), connectionstyle="arc3,rad=-0.2")
    
    # Quantum Circuit 2 (QNN1)
    qnn2_x, qnn2_y = 8.5, 0.6
    draw_quantum_circuit_box(ax, qnn2_x, qnn2_y, 1.4, 0.6)
    ax.text(qnn2_x, qnn2_y + 0.1, 'QUEEN', ha='center', fontsize=7, 
            fontweight='bold', color='purple')
    ax.text(qnn2_x, qnn2_y - 0.1, '4Q × 1L', ha='center', fontsize=6)
    draw_arrow(ax, (8.5, 1.15), (8.5, 0.9))
    
    # === FUSION & OUTPUT ===
    draw_rounded_box(ax, 7.25, -0.3, 1.8, 0.5, 'Attention Fusion\n(W_att · [Classical, Quantum])', '#FFB6C1', 7)
    draw_arrow(ax, (6, 0.5), (6.75, -0.05), connectionstyle="arc3,rad=0.2")
    draw_arrow(ax, (8.5, 0.3), (7.75, -0.05), connectionstyle="arc3,rad=-0.2")
    
    draw_rounded_box(ax, 7.25, -1.0, 1.4, 0.5, 'Final Linear\n(4→2) + Softmax', '#98FB98', 8)
    draw_arrow(ax, (7.25, -0.55), (7.25, -0.75))
    
    # Output
    draw_rounded_box(ax, 7.25, -1.7, 1.6, 0.5, 'Output Y\n(Change Probability)', '#FFD700', 9)
    draw_arrow(ax, (7.25, -1.25), (7.25, -1.45))
    
    # === LEGEND ===
    legend_y = -0.5
    legend_elements = [
        (0.5, '#FFE4B5', 'Input Data'),
        (1.8, '#87CEEB', 'Conv Layers'),
        (3.1, '#90EE90', 'GAT/Graph'),
        (4.4, '#E6E6FA', 'Quantum'),
        (5.7, '#FFA07A', 'Activation'),
        (7.0, '#FFB6C1', 'Fusion'),
    ]
    
    for x, color, label in legend_elements:
        box = Rectangle((x - 0.2, legend_y - 0.1), 0.4, 0.2, 
                        facecolor=color, edgecolor='black', linewidth=0.5)
        ax.add_patch(box)
        ax.text(x + 0.35, legend_y, label, fontsize=7, va='center')
    
    # === ANNOTATIONS ===
    # Quantum circuit description
    ax.text(9.8, 3.5, 'QUEEN Circuit:', ha='left', fontsize=8, fontweight='bold')
    ax.text(9.8, 3.2, '• RY angle encoding', ha='left', fontsize=7)
    ax.text(9.8, 2.95, '• IsingXX entanglement', ha='left', fontsize=7)
    ax.text(9.8, 2.7, '• RX/RY trainable gates', ha='left', fontsize=7)
    ax.text(9.8, 2.45, '• Multi-controlled X', ha='left', fontsize=7)
    ax.text(9.8, 2.2, '• PauliZ measurements', ha='left', fontsize=7)
    
    plt.tight_layout()
    return fig, ax

if __name__ == "__main__":
    import os
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate QDN architecture diagram')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for diagrams (default: docs/)')
    args = parser.parse_args()
    
    # Determine output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(repo_root, 'docs')
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = create_qdn_architecture()
    
    # Save as PNG
    output_path = os.path.join(output_dir, 'qdn_architecture.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"Architecture diagram saved to: {output_path}")
    
    # Also save as PDF
    pdf_path = os.path.join(output_dir, 'qdn_architecture.pdf')
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"PDF version saved to: {pdf_path}")
    
    plt.close()
