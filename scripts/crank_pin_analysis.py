#!/usr/bin/env python3
"""
Crank-Pin Bearing Analysis Script
Calculates bearing reactions and pressure for engine crank mechanism.
Converted from MATLAB code for mechanical engineering analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

# Input geometry & weights
rpm = 1500                      # engine speed
omega = 2 * np.pi * (rpm / 60)  # rad/s
stroke = 180e-3                 # m (given)
r = stroke / 2                  # crank radius (m)
l_over_r = 4                    # l/r ratio (given)
l = l_over_r * r                # connecting rod length (m)
g = 9.81                        # gravity m/s^2

# Weights (N) as per sheet
W_piston = 50       # weight of piston + pin + ring etc (N)
W_big_end = 30      # big end weight (N)
W_shank = 6         # weight of shank acting at distance a from big end (N)
W_small_end = 5     # small end weight (N)

# Distances for lumped mass conversion
a_mm = 80           # a = 80 mm (distance of shank from big end centre)
L_mm = 360          # total rod length (a + b)
a = a_mm / 1000     # m
L = L_mm / 1000     # m
b = L - a           # m

# Crank-pin geometry for pressure calculation
D_pin = 112e-3      # crank pin diameter (m)
L_pin = 56e-3       # crank pin length (m)

# Convert connecting-rod shank weight into two lumped masses
WCB = W_shank * (b / (a + b))   # weight lumped at big end
WCS = W_shank * (a / (a + b))   # weight lumped at small end

# Total reciprocating and rotating weights (weights are in N)
W_rec = W_small_end + WCS + W_piston  # total reciprocating weight (N)
W_rot = W_big_end + WCB               # total rotating weight (N)

print(f'Lumped weights: W_CS = {WCS:.3f} N, W_CB = {WCB:.3f} N')
print(f'W_rec = {W_rec:.3f} N, W_rot = {W_rot:.3f} N\n')

# Gas force data (approximate values from table)
known_angles = np.array([0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 
                        240, 270, 300, 330, 360, 420, 480, 540, 600, 660, 720])
known_gas_kN = np.array([65, 85, 66, 24, 13, 9, 7.5, 4.5, 2.1, 0.25, 0.25, 0.25, 
                        0.25, 0.25, 0.25, -0.1, 0.01, 0.6, 2, 6, 30, 60, 65])

# Build high-resolution grid
theta_all = np.arange(0, 721, 1)  # degrees (0 to 720)

# Interpolate gas force to 0:1:720 deg using PCHIP
interpolator = PchipInterpolator(known_angles, known_gas_kN)
gas_kN_interp = interpolator(theta_all)

# Convert to N
gas_N = gas_kN_interp * 1e3

# Inertia force of reciprocating mass (Fi)
n = l_over_r  # n = l/r
Fi_amp = (W_rec * omega**2 * r) / g  # amplitude coefficient, unit N

theta_rad = np.deg2rad(theta_all)
Fi = Fi_amp * (np.cos(theta_rad) + (np.cos(2 * theta_rad)) / n)  # N

# Constant centrifugal force of rotating mass (Fcr)
F_cr = (W_rot * omega**2 * r) / g  # N (constant for all theta)

# Net reciprocating force Fr (along cylinder axis)
Fr = gas_N + Fi  # N

# Connecting rod angle phi (radians)
sin_phi = (r / l) * np.sin(theta_rad)
# Limit to [-1, 1] to avoid numeric errors
sin_phi = np.clip(sin_phi, -1, 1)
phi = np.arcsin(sin_phi)  # radians

# Thrust in connecting rod T
T = Fr / np.cos(phi)  # N

# Resultant bearing reaction at crank pin, FR(theta)
FR = np.sqrt(T**2 + F_cr**2 + 2 * T * F_cr * np.cos(theta_rad + phi))  # N

# Numeric integration to get average reaction over full crank rotation (0..720 deg)
from scipy.integrate import trapezoid
F_avg = trapezoid(FR, theta_all) / 720  # N

# Average bearing pressure p_av = (F_avg) / (projected bearing area L_pin * D_pin)
bearing_area = L_pin * D_pin  # m^2
p_av_Pa = F_avg / bearing_area  # Pa
p_av_MPa = p_av_Pa / 1e6

# Display key results
print(f'Constant centrifugal force F_cr = {F_cr/1e3:.2f} kN')
print(f'Average crank-pin resultant force Favg = {F_avg:.2f} N ({F_avg/1e3:.3f} kN)')
print(f'Average crank-pin pressure p_av = {p_av_MPa:.3f} MPa ({p_av_MPa:.1f} MPa = {p_av_Pa/1e6:.0f} N/mm^2)\n')

# Plot FR vs crank angle
fig = plt.figure(figsize=(10, 6))
plt.plot(theta_all, FR / 1e3, linewidth=1.6)
plt.xlabel('Crank angle θ (degrees)')
plt.ylabel('Resultant bearing reaction F_R (kN)')
plt.title('Crank-pin resultant reaction F_R vs crank angle θ')
plt.grid(True)
plt.xlim([0, 720])

# Mark average force line
plt.axhline(y=F_avg / 1e3, color='r', linestyle='--', linewidth=1.2, 
            label=f'F_avg = {F_avg/1e3:.2f} kN')
plt.legend(loc='upper left')

# Save the plot
output_filename = 'crank_pin_bearing_analysis.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f'Plot saved to: {output_filename}')

# Show intermediate table for certain step sizes
check_angles = np.arange(0, 721, 20)
FR_check = np.interp(check_angles, theta_all, FR)

print('\nIntermediate values (every 20 degrees):')
print(f'{"theta_deg":>10s} {"FR_kN":>10s}')
print('-' * 22)
for angle, fr_val in zip(check_angles, FR_check):
    print(f'{angle:10.0f} {fr_val/1e3:10.3f}')

print('\nAnalysis complete!')
