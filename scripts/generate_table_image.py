#!/usr/bin/env python3
"""
Generate a clean PNG image of the Crank Pin Load table
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Data from the analysis
theta = np.array([0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 
                  270, 300, 330, 360, 390, 420, 450, 480, 510, 540, 570, 600, 
                  630, 660, 690, 720])

Fg = np.array([65, 85, 66, 24, 13, 9, 7.5, 4.5, 2.1, 0.25, 0.25, 0.25, 0.25, 
               0.25, 0.25, 0.25, 0.25, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, 
               -0.10, -0.01, 6.6, 2.0, 6.0, 30, 65]) * 1e3

# Constants
m_big = 3.534  # kg
a_big = 2229.67  # m/s^2

# Calculations
theta_rad = np.deg2rad(theta)
phi = np.arcsin(np.sin(theta_rad) / 4)
Q = (-12.760) * (np.cos(theta_rad) + np.cos(2*theta_rad)/4 + Fg[:len(theta)]) / np.cos(phi)
Fcr = m_big * a_big
Fpl = np.sqrt(Q**2 + Fcr**2 - 2 * Q * Fcr * np.cos(theta_rad + phi))
Fpl_kN = Fpl / 1000
Fpl_avg = np.mean(Fpl)

# Create figure
fig, ax = plt.subplots(figsize=(8, 12))
ax.axis('tight')
ax.axis('off')

# Prepare table data
table_data = []
for i in range(len(theta)):
    table_data.append([f"{int(theta[i])}", f"{Fpl_kN[i]:.2f}"])

# Create table
table = ax.table(cellText=table_data,
                colLabels=['Crank Angle (°)', 'F_PL (kN)'],
                cellLoc='center',
                loc='center',
                colWidths=[0.4, 0.4])

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2)

# Style header
for i in range(2):
    cell = table[(0, i)]
    cell.set_facecolor('#4472C4')
    cell.set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(theta) + 1):
    for j in range(2):
        cell = table[(i, j)]
        if i % 2 == 0:
            cell.set_facecolor('#E7E6E6')
        else:
            cell.set_facecolor('white')

# Add title
plt.title('Resultant Crank Pin Load vs Crank Angle', 
          fontsize=16, fontweight='bold', pad=20)

# Add average value at bottom
fig.text(0.5, 0.05, f'Average F_PL = {Fpl_avg/1000:.2f} kN', 
         ha='center', fontsize=13, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='#FFD966', alpha=0.8))

# Save figure
plt.savefig('/home/runner/work/maneet/maneet/scripts/crank_pin_load_table.png', 
            dpi=150, bbox_inches='tight', facecolor='white')
print("Table image saved as: crank_pin_load_table.png")
print(f"\nAverage Resultant Crank Pin Load F_PL(avg) = {Fpl_avg/1000:.2f} kN")
