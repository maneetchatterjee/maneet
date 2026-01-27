#!/usr/bin/env python3
"""
Bearing Analysis Script
===========Maneet Chatterjee============
===========2023MEB045============

This script performs bearing analysis calculations and generates output tables and plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Create output directory
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output')
os.makedirs(output_dir, exist_ok=True)

# Input parameters
Ti = 80                  # °C
rho = 860                # kg/m^3
Cp = 1700                # J/kg°C
pav = 2.96e6             # Pa

N = 1500/60              # rps
D = 112                  # mm
L = 56e-3                # m

To = np.array([105, 107, 109, 112, 116, 127], dtype=float)
eta = np.array([8.5, 8.2, 7.9, 7.4, 6.9, 5.8]) * 1e-3

lambda0 = 9.80
lambda1 = 15.0
lambda2 = 26.0
S0 = 0.0314
S1 = 0.0921
S2 = 0.3210

# Calculations
TB = 0.5 * (To + Ti)
dT = (To - Ti) / 0.8

lambda_val = (rho * Cp * dT) / pav

# Calculate S using interpolation formula
S = np.zeros_like(lambda_val)
for i in range(len(lambda_val)):
    S[i] = (S0 
            + 0.011673 * (lambda_val[i] - lambda0)
            + 5.64197e-4 * (lambda_val[i] - lambda0) * (lambda_val[i] - lambda1)
            - 0.09452e-4 * (lambda_val[i] - lambda0) * (lambda_val[i] - lambda1) * (lambda_val[i] - lambda2))

DbyC = np.sqrt((S * pav) / (eta * N))  # D/C
C = D / DbyC                            # mm

twohbyC = (0.2 
           + 3.2949 * (S - S0)
           - 5.3432 * (S - S0) * (S - S1)
           + 5.3218 * (S - S0) * (S - S1) * (S - S2))

h0 = (twohbyC * C) / 2

# Create TABLE IIA
Table_IIA = pd.DataFrame({
    'To(C)': To,
    'eta(Pa.s)': eta,
    'TB(C)': TB,
    'DeltaT': dT,
    'lambda': lambda_val,
    'S': S,
    'D/C': DbyC,
    'C(mm)': C,
    '2h0/C': twohbyC,
    'h0_mm': h0
})

print('TABLE IIA')
print(Table_IIA.to_string(index=False))
print()

# Calculate Q
QS = (3.17 
      + 6.42504 * (S - S0)
      - 16.6043 * (S - S0) * (S - S1)
      + 21.82871 * (S - S0) * (S - S1) * (S - S2))

Q = (QS * D * C * 1e-3 * N * L) / 2  # m^3/min

# Create TABLE IIB
Table_IIB = pd.DataFrame({
    'S': S,
    'C_mm': C,
    '2Q/DCNL': QS,
    'Q(m3/min)': Q,
    'TB(C)': TB,
    'h0(mm)': h0
})

print('TABLE IIB')
print(Table_IIB.to_string(index=False))

# Save tables to file
table_output_path = os.path.join(output_dir, 'tables_output.txt')
with open(table_output_path, 'w') as f:
    f.write('TABLE IIA\n')
    f.write(Table_IIA.to_string(index=False))
    f.write('\n\n')
    f.write('TABLE IIB\n')
    f.write(Table_IIB.to_string(index=False))
    f.write('\n')

print(f'\nTables saved to: {table_output_path}')

# Also save as CSV files
Table_IIA.to_csv(os.path.join(output_dir, 'table_IIA.csv'), index=False)
Table_IIB.to_csv(os.path.join(output_dir, 'table_IIB.csv'), index=False)
print(f'CSV files saved to: {output_dir}')

# Plot 1: TB vs C
plt.figure(figsize=(8, 6))
plt.plot(C, TB, '-o', linewidth=1.5)
plt.grid(True)
plt.xlabel('Diametral Clearance, C (mm)')
plt.ylabel('Bearing Mean Temperature, T_B (°C)')
plt.title('T_B vs C')
plt.tight_layout()
plot1_path = os.path.join(output_dir, 'TB_vs_C.png')
plt.savefig(plot1_path, dpi=150)
print(f'Plot saved to: {plot1_path}')

# Plot 2: h0 vs C
plt.figure(figsize=(8, 6))
plt.plot(C, h0, '-o', linewidth=1.5)
plt.grid(True)
plt.xlabel('Diametral Clearance, C (mm)')
plt.ylabel('Minimum Film Thickness, h_0 (mm)')
plt.title('h_0 vs C')
plt.tight_layout()
plot2_path = os.path.join(output_dir, 'h0_vs_C.png')
plt.savefig(plot2_path, dpi=150)
print(f'Plot saved to: {plot2_path}')

# Plot 3: Q vs C
plt.figure(figsize=(8, 6))
plt.plot(C, Q, '-o', linewidth=1.5)
plt.grid(True)
plt.xlabel('Diametral Clearance, C (mm)')
plt.ylabel('Oil Flow Rate, Q (m³/min)')
plt.title('Q vs C')
plt.tight_layout()
plot3_path = os.path.join(output_dir, 'Q_vs_C.png')
plt.savefig(plot3_path, dpi=150)
print(f'Plot saved to: {plot3_path}')

print('\nAll outputs generated successfully!')
