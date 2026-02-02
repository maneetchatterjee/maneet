"""
Generate Architecture Diagram

Creates a visual representation of the VLA Pipeline architecture.
Requires matplotlib.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def create_architecture_diagram():
    """Create a visual architecture diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'VLA Pipeline Architecture', 
            ha='center', va='top', fontsize=20, fontweight='bold')
    
    # Define colors
    perception_color = '#FFE5E5'
    language_color = '#E5F5FF'
    planning_color = '#E5FFE5'
    control_color = '#FFF5E5'
    simulation_color = '#F5E5FF'
    pipeline_color = '#EFEFEF'
    
    # Main Pipeline Box
    pipeline_box = FancyBboxPatch((0.5, 0.5), 9, 8,
                                  boxstyle="round,pad=0.1",
                                  edgecolor='black', facecolor=pipeline_color,
                                  linewidth=2, alpha=0.3)
    ax.add_patch(pipeline_box)
    
    # Module boxes
    modules = [
        # (x, y, width, height, color, name)
        (1, 6.5, 2, 1.5, perception_color, 'Perception\nModule'),
        (3.5, 6.5, 2, 1.5, language_color, 'Language\nModule'),
        (6, 6.5, 2, 1.5, planning_color, 'Planning\nModule'),
        (4, 4.5, 2, 1.5, control_color, 'Control\nModule'),
        (3, 2, 4, 1.5, simulation_color, 'Simulation\nEnvironment'),
    ]
    
    for x, y, w, h, color, name in modules:
        box = FancyBboxPatch((x, y), w, h,
                             boxstyle="round,pad=0.05",
                             edgecolor='black', facecolor=color,
                             linewidth=2)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, name,
                ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Add module descriptions
    descriptions = [
        (2, 6.2, 'CV-based\nDetection'),
        (4.5, 6.2, 'NLP Command\nParsing'),
        (7, 6.2, 'Symbolic\nPlanning'),
        (5, 4.2, 'IK/FK\nController'),
        (5, 1.7, 'PyBullet Physics'),
    ]
    
    for x, y, text in descriptions:
        ax.text(x, y, text, ha='center', va='top', fontsize=8, style='italic')
    
    # Arrows showing data flow
    arrows = [
        # (x1, y1, x2, y2, label)
        (2, 6.5, 3.5, 6.5, 'Scene\nObjects'),
        (4.5, 6.5, 6, 6.5, 'Parsed\nCommand'),
        (7, 6.5, 6, 4.5, 'Waypoints'),
        (5, 4.5, 5, 3.5, 'Joint\nAngles'),
        (7, 2.75, 8, 6.5, 'Feedback'),
        (2, 2.75, 1.5, 6.5, 'RGB-D\nImages'),
    ]
    
    for x1, y1, x2, y2, label in arrows:
        if 'Feedback' in label:
            arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                   arrowstyle='->', mutation_scale=20,
                                   linewidth=2, color='green', alpha=0.7,
                                   connectionstyle="arc3,rad=.3")
        elif 'Images' in label:
            arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                   arrowstyle='->', mutation_scale=20,
                                   linewidth=2, color='blue', alpha=0.7,
                                   connectionstyle="arc3,rad=-.3")
        else:
            arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                   arrowstyle='->', mutation_scale=20,
                                   linewidth=2, color='black', alpha=0.7)
        ax.add_patch(arrow)
        
        # Add label
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        if 'Feedback' in label or 'Images' in label:
            offset = 0.3
        else:
            offset = 0
        ax.text(mid_x + offset, mid_y, label,
                ha='center', va='bottom', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add input/output
    ax.text(0.8, 8.5, 'Input:\nNatural Language\nCommand',
            ha='left', va='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    ax.text(9.2, 8.5, 'Output:\nTask Execution\n+ Metrics',
            ha='right', va='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # Add legend
    legend_y = 0.3
    ax.text(1, legend_y, '⬤ Modular Design  ⬤ Extensible  ⬤ Simulation-First',
            ha='left', va='center', fontsize=9, style='italic')
    
    plt.tight_layout()
    
    # Save diagram
    output_path = os.path.join(os.path.dirname(__file__), '..', 'docs', 'architecture_diagram.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Architecture diagram saved to: {output_path}")
    
    # Also save as PDF
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"PDF version saved to: {pdf_path}")


if __name__ == "__main__":
    try:
        create_architecture_diagram()
        print("\n✓ Architecture diagram generated successfully!")
    except ImportError as e:
        print(f"Error: {e}")
        print("Please install matplotlib: pip install matplotlib")
    except Exception as e:
        print(f"Error generating diagram: {e}")
        import traceback
        traceback.print_exc()
