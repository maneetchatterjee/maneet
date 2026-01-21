#!/usr/bin/env python3
"""
Generate QDN (Quantum Detection Network) architecture diagram.

This script creates a clean, research-paper-ready visualization of the hybrid 
quantum-classical neural network architecture for change detection in 
hyperspectral images.

The diagram is generated using matplotlib with a professional aesthetic suitable
for academic publications. A draw.io file is also provided for further editing.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Polygon
import matplotlib.patheffects as pe
import numpy as np

# Color scheme - professional and publication-ready
COLORS = {
    'input': '#FFF3E0',        # Light orange
    'conv': '#E3F2FD',         # Light blue
    'gat': '#E8F5E9',          # Light green
    'quantum': '#F3E5F5',      # Light purple
    'activation': '#FFF8E1',   # Light amber
    'fusion': '#FCE4EC',       # Light pink
    'output': '#FFFDE7',       # Light yellow
    'border_input': '#E65100',
    'border_conv': '#1565C0',
    'border_gat': '#2E7D32',
    'border_quantum': '#7B1FA2',
    'border_activation': '#FF8F00',
    'border_fusion': '#C2185B',
    'border_output': '#F9A825',
}


def draw_box(ax, x, y, width, height, text, fill_color, border_color, 
             fontsize=9, fontweight='normal', linestyle='-', linewidth=1.5):
    """Draw a rounded rectangle box with text."""
    box = FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=fill_color, edgecolor=border_color, 
        linewidth=linewidth, linestyle=linestyle
    )
    ax.add_patch(box)
    
    # Handle multi-line text
    lines = text.split('\n')
    line_height = 0.12
    start_y = y + (len(lines) - 1) * line_height / 2
    
    for i, line in enumerate(lines):
        ax.text(x, start_y - i * line_height, line, 
                ha='center', va='center', fontsize=fontsize,
                fontweight=fontweight, color='#212121',
                family='DejaVu Sans')


def draw_arrow(ax, start, end, color='#424242', style='-|>', 
               connectionstyle="arc3,rad=0", linewidth=1.2):
    """Draw an arrow between two points."""
    arrow = FancyArrowPatch(
        start, end, arrowstyle=style, color=color,
        connectionstyle=connectionstyle, linewidth=linewidth,
        mutation_scale=12, shrinkA=2, shrinkB=2
    )
    ax.add_patch(arrow)


def draw_group_box(ax, x, y, width, height, title, fill_color, border_color):
    """Draw a group box with title."""
    # Main box
    box = FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle="round,pad=0.03,rounding_size=0.1",
        facecolor=fill_color, edgecolor=border_color,
        linewidth=2, alpha=0.3
    )
    ax.add_patch(box)
    
    # Title
    ax.text(x, y + height/2 - 0.15, title, 
            ha='center', va='center', fontsize=9,
            fontweight='bold', color=border_color,
            family='DejaVu Sans')


def create_qdn_architecture():
    """Create the complete QDN architecture diagram - research paper ready."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(-0.5, 12.5)
    ax.set_ylim(-2.5, 8.5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # White background
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # === TITLE ===
    ax.text(6, 8.2, 'QDN: Quantum Detection Network', 
            ha='center', va='center', fontsize=14, fontweight='bold',
            family='DejaVu Sans', color='#212121')
    ax.text(6, 7.85, 'Hybrid Quantum-Classical Architecture for Hyperspectral Change Detection', 
            ha='center', va='center', fontsize=10, style='italic', 
            color='#616161', family='DejaVu Sans')
    
    # === INPUT SECTION ===
    draw_box(ax, 1.0, 7.0, 1.1, 0.55, 'X₁\n(H×W×C)', 
             COLORS['input'], COLORS['border_input'], fontsize=9, fontweight='bold')
    draw_box(ax, 2.5, 7.0, 1.1, 0.55, 'X₂\n(H×W×C)', 
             COLORS['input'], COLORS['border_input'], fontsize=9, fontweight='bold')
    draw_box(ax, 4.0, 7.0, 1.1, 0.55, 'Spectral\nMap', 
             COLORS['gat'], COLORS['border_gat'], fontsize=9, fontweight='bold')
    
    # === DIMENSION REDUCTION ===
    draw_box(ax, 1.0, 6.0, 1.0, 0.45, 'Conv 1×1\n(C→64)', 
             COLORS['conv'], COLORS['border_conv'], fontsize=8)
    draw_box(ax, 2.5, 6.0, 1.0, 0.45, 'Conv 1×1\n(C→64)', 
             COLORS['conv'], COLORS['border_conv'], fontsize=8)
    
    draw_arrow(ax, (1.0, 6.72), (1.0, 6.23))
    draw_arrow(ax, (2.5, 6.72), (2.5, 6.23))
    
    # PReLU
    draw_box(ax, 1.0, 5.4, 0.7, 0.3, 'PReLU', 
             COLORS['activation'], COLORS['border_activation'], fontsize=7)
    draw_box(ax, 2.5, 5.4, 0.7, 0.3, 'PReLU', 
             COLORS['activation'], COLORS['border_activation'], fontsize=7)
    
    draw_arrow(ax, (1.0, 5.77), (1.0, 5.55))
    draw_arrow(ax, (2.5, 5.77), (2.5, 5.55))
    
    # === DIFFERENCE ===
    draw_box(ax, 1.75, 4.7, 1.3, 0.45, 'Xdiff = X₁ − X₂', 
             COLORS['fusion'], COLORS['border_fusion'], fontsize=8, fontweight='bold')
    
    draw_arrow(ax, (1.0, 5.25), (1.4, 4.93), connectionstyle="arc3,rad=-0.2")
    draw_arrow(ax, (2.5, 5.25), (2.1, 4.93), connectionstyle="arc3,rad=0.2")
    
    # === GRAPH ATTENTION BRANCH ===
    draw_group_box(ax, 1.0, 3.3, 1.8, 2.0, 'Graph Attention', 
                   COLORS['gat'], COLORS['border_gat'])
    
    draw_box(ax, 1.0, 3.9, 1.4, 0.4, 'Superpixel Pool\nQᵀ · Xdiff', 
             COLORS['gat'], COLORS['border_gat'], fontsize=7)
    draw_box(ax, 1.0, 3.2, 1.4, 0.5, 'GAT\n(64→128→64)\n2-Head', 
             COLORS['gat'], COLORS['border_gat'], fontsize=7, fontweight='bold')
    draw_box(ax, 1.0, 2.55, 1.4, 0.35, 'Q · H', 
             COLORS['gat'], COLORS['border_gat'], fontsize=7)
    
    draw_arrow(ax, (1.4, 4.47), (1.0, 4.1), connectionstyle="arc3,rad=0.2")
    draw_arrow(ax, (1.0, 3.7), (1.0, 3.45))
    draw_arrow(ax, (1.0, 2.95), (1.0, 2.73))
    
    # === QFL BRANCH ===
    draw_group_box(ax, 3.5, 3.3, 2.0, 2.0, 'Quantum Feature Learning', 
                   COLORS['quantum'], COLORS['border_quantum'])
    
    draw_box(ax, 3.5, 3.9, 1.3, 0.4, 'Conv 1×1\n(64→8)', 
             COLORS['conv'], COLORS['border_conv'], fontsize=7)
    draw_box(ax, 3.5, 3.2, 1.6, 0.5, 'QUEEN\n4 Qubits × 1 Layer', 
             COLORS['quantum'], COLORS['border_quantum'], fontsize=7, 
             fontweight='bold', linestyle='--', linewidth=2)
    draw_box(ax, 3.5, 2.55, 1.3, 0.35, 'Conv 1×1\n(4→64)', 
             COLORS['conv'], COLORS['border_conv'], fontsize=7)
    
    draw_arrow(ax, (2.1, 4.47), (3.5, 4.1), connectionstyle="arc3,rad=-0.1")
    draw_arrow(ax, (3.5, 3.7), (3.5, 3.45))
    draw_arrow(ax, (3.5, 2.95), (3.5, 2.73))
    
    # === FEATURE FUSION ===
    draw_box(ax, 2.25, 1.8, 2.8, 0.45, 'Feature Fusion: (GAT + QFL) × Spectral', 
             COLORS['fusion'], COLORS['border_fusion'], fontsize=8, fontweight='bold')
    
    draw_arrow(ax, (1.0, 2.37), (1.5, 2.03), connectionstyle="arc3,rad=0.1")
    draw_arrow(ax, (3.5, 2.37), (3.0, 2.03), connectionstyle="arc3,rad=-0.1")
    
    # Spectral map connection
    draw_arrow(ax, (4.0, 6.72), (4.0, 5.5), color='#616161', style='->', linewidth=1)
    ax.plot([4.0, 4.0], [5.5, 1.8], 'k--', linewidth=0.8, alpha=0.5)
    draw_arrow(ax, (4.0, 1.95), (3.65, 1.9), connectionstyle="arc3,rad=0.2", 
               color='#616161', linewidth=1)
    
    # === CONCATENATION ===
    draw_box(ax, 6.0, 5.5, 1.8, 0.5, 'Concatenate\n[F, X₁, X₂, S] → 193ch', 
             COLORS['input'], COLORS['border_input'], fontsize=7)
    
    draw_arrow(ax, (2.25, 1.57), (5.1, 5.35), connectionstyle="arc3,rad=-0.15")
    
    # === FEATURE EXTRACTION ===
    draw_group_box(ax, 6.0, 4.0, 2.0, 2.4, 'Feature Extraction', 
                   COLORS['conv'], COLORS['border_conv'])
    
    draw_box(ax, 6.0, 4.7, 1.5, 0.4, 'Conv 3×3\n(193→64)', 
             COLORS['conv'], COLORS['border_conv'], fontsize=7)
    draw_box(ax, 6.0, 4.15, 1.2, 0.3, 'BN + PReLU', 
             COLORS['activation'], COLORS['border_activation'], fontsize=7)
    draw_box(ax, 6.0, 3.6, 1.5, 0.4, 'Conv 3×3\n(64→32)', 
             COLORS['conv'], COLORS['border_conv'], fontsize=7)
    draw_box(ax, 6.0, 3.05, 1.2, 0.3, 'BN + PReLU', 
             COLORS['activation'], COLORS['border_activation'], fontsize=7)
    
    draw_arrow(ax, (6.0, 5.25), (6.0, 4.9))
    draw_arrow(ax, (6.0, 4.5), (6.0, 4.3))
    draw_arrow(ax, (6.0, 4.0), (6.0, 3.8))
    draw_arrow(ax, (6.0, 3.4), (6.0, 3.2))
    
    # === DUAL CLASSIFICATION ===
    # Classical
    draw_group_box(ax, 8.0, 3.8, 1.6, 1.4, 'Classical', 
                   COLORS['conv'], COLORS['border_conv'])
    draw_box(ax, 8.0, 3.9, 1.2, 0.4, 'Linear\n(32→2)', 
             COLORS['conv'], COLORS['border_conv'], fontsize=7)
    draw_box(ax, 8.0, 3.35, 1.0, 0.3, 'Softmax', 
             COLORS['gat'], COLORS['border_gat'], fontsize=7)
    
    draw_arrow(ax, (6.75, 3.05), (7.4, 3.9), connectionstyle="arc3,rad=-0.1")
    draw_arrow(ax, (8.0, 3.7), (8.0, 3.5))
    
    # QEC
    draw_group_box(ax, 8.0, 2.0, 1.8, 1.5, 'Quantum Classification', 
                   COLORS['quantum'], COLORS['border_quantum'])
    draw_box(ax, 8.0, 2.2, 1.3, 0.35, 'Conv 1×1\n(32→4)', 
             COLORS['conv'], COLORS['border_conv'], fontsize=7)
    draw_box(ax, 8.0, 1.65, 1.5, 0.4, 'QUEEN\n4Q × 1L', 
             COLORS['quantum'], COLORS['border_quantum'], fontsize=7, 
             fontweight='bold', linestyle='--', linewidth=2)
    
    draw_arrow(ax, (6.75, 2.9), (7.35, 2.2), connectionstyle="arc3,rad=0.1")
    draw_arrow(ax, (8.0, 2.02), (8.0, 1.85))
    
    # === ATTENTION FUSION ===
    draw_box(ax, 10.0, 2.8, 2.0, 0.5, 'Attention Fusion\nWatt · [C, Q]', 
             COLORS['fusion'], COLORS['border_fusion'], fontsize=8, fontweight='bold')
    
    draw_arrow(ax, (8.5, 3.35), (9.0, 2.95), connectionstyle="arc3,rad=-0.1")
    draw_arrow(ax, (8.75, 1.65), (9.0, 2.65), connectionstyle="arc3,rad=0.2")
    
    # === FINAL OUTPUT ===
    draw_box(ax, 10.0, 2.0, 1.5, 0.4, 'Linear (4→2)', 
             COLORS['conv'], COLORS['border_conv'], fontsize=8)
    draw_box(ax, 10.0, 1.45, 1.2, 0.3, 'Softmax', 
             COLORS['gat'], COLORS['border_gat'], fontsize=8)
    draw_box(ax, 10.0, 0.8, 1.8, 0.5, 'Output Y\nΔ Probability', 
             COLORS['output'], COLORS['border_output'], fontsize=9, fontweight='bold')
    
    draw_arrow(ax, (10.0, 2.55), (10.0, 2.2))
    draw_arrow(ax, (10.0, 1.8), (10.0, 1.6))
    draw_arrow(ax, (10.0, 1.3), (10.0, 1.05))
    
    # === QUEEN CIRCUIT DETAIL ===
    detail_x = 10.5
    detail_y = 5.8
    
    # Box for circuit details
    detail_box = FancyBboxPatch(
        (detail_x - 1.3, detail_y - 1.6), 2.6, 2.8,
        boxstyle="round,pad=0.03,rounding_size=0.1",
        facecolor='#FAFAFA', edgecolor='#9E9E9E',
        linewidth=1.5
    )
    ax.add_patch(detail_box)
    
    ax.text(detail_x, detail_y + 1.0, 'QUEEN Circuit', 
            ha='center', va='center', fontsize=9, fontweight='bold',
            color=COLORS['border_quantum'])
    
    circuit_info = [
        '① RY(data) - Angle encoding',
        '② RY(θ) - Trainable rotation',
        '③ IsingXX - Entanglement',
        '④ RX(θ) - Trainable rotation',
        '⑤ IsingXX - Entanglement',
        '⑥ RY(θ) - Trainable rotation',
        '⑦ MCX - Multi-controlled X',
        '⑧ ⟨Z⟩ - PauliZ measurement',
    ]
    
    for i, info in enumerate(circuit_info):
        ax.text(detail_x - 1.1, detail_y + 0.6 - i * 0.25, info, 
                ha='left', va='center', fontsize=7,
                color='#424242', family='DejaVu Sans')
    
    # === LEGEND ===
    legend_x = 0.3
    legend_y = 0.3
    legend_items = [
        (COLORS['input'], COLORS['border_input'], 'Input/Output'),
        (COLORS['conv'], COLORS['border_conv'], 'Convolution'),
        (COLORS['gat'], COLORS['border_gat'], 'Graph Attention'),
        (COLORS['quantum'], COLORS['border_quantum'], 'Quantum Circuit'),
        (COLORS['fusion'], COLORS['border_fusion'], 'Feature Fusion'),
    ]
    
    for i, (fill, border, label) in enumerate(legend_items):
        box = Rectangle((legend_x + i * 2.2, legend_y), 0.3, 0.2, 
                        facecolor=fill, edgecolor=border, linewidth=1.5)
        ax.add_patch(box)
        ax.text(legend_x + i * 2.2 + 0.4, legend_y + 0.1, label, 
                fontsize=7, va='center', color='#424242')
    
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
    
    # Save as PNG (high resolution for papers)
    output_path = os.path.join(output_dir, 'qdn_architecture.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"Architecture diagram saved to: {output_path}")
    
    # Also save as PDF (vector format for papers)
    pdf_path = os.path.join(output_dir, 'qdn_architecture.pdf')
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"PDF version saved to: {pdf_path}")
    
    print("\nNote: A draw.io file (qdn_architecture.drawio) is also available")
    print("in the docs/ directory for further editing.")
    
    plt.close()
