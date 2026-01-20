# Crank Pin Load Analysis

## Overview
This analysis calculates the resultant crank pin load (F_PL) as a function of crank angle for a reciprocating engine mechanism. The analysis covers two complete rotations (0° to 720°).

## Files Generated
1. **crank_pin_load_analysis.m** - MATLAB/Octave script containing all calculations
2. **crank_pin_load_plot.png** - Graphical representation of F_PL vs Crank Angle
3. **crank_pin_load_table.csv** - Tabular data with all calculated values

## How to Run
```bash
# Using Octave (open-source MATLAB alternative)
octave --no-gui crank_pin_load_analysis.m

# Or using MATLAB
matlab -batch "run('crank_pin_load_analysis.m')"
```

## Results

### Average Resultant Crank Pin Load
**F_PL(avg) = 147.37 kN**

### Complete Table of Values

| Crank Angle (°) | F_PL (kN) |
|-----------------|-----------|
| 0               | 837.30    |
| 20              | 1095.75   |
| 40              | 858.42    |
| 60              | 316.14    |
| 80              | 170.74    |
| 100             | 115.47    |
| 120             | 92.87     |
| 140             | 51.55     |
| 160             | 19.39     |
| 180             | 4.70      |
| 200             | 4.87      |
| 220             | 5.37      |
| 240             | 6.17      |
| 270             | 7.74      |
| 300             | 9.40      |
| 330             | 10.63     |
| 360             | 11.09     |
| 390             | 6.91      |
| 420             | 7.59      |
| 450             | 8.31      |
| 480             | 8.82      |
| 510             | 9.09      |
| 540             | 9.17      |
| 570             | 9.09      |
| 600             | 7.97      |
| 630             | 85.35     |
| 660             | 29.49     |
| 690             | 83.59     |
| 720             | 390.70    |

## Analysis Summary

### Key Observations
- **Maximum Load**: 1095.75 kN at 20° crank angle
- **Minimum Load**: 4.70 kN at 180° crank angle
- **Load Range**: 1091.05 kN difference between max and min
- **Cyclic Pattern**: The load exhibits a strong cyclic behavior with peaks in the early portion of each rotation

### Physical Interpretation
The resultant crank pin load shows:
1. **High loads during compression/power stroke** (0°-60°)
2. **Very low loads at mid-stroke** (180°-540°)
3. **Rising loads as cycle repeats** (630°-720°)

This pattern is characteristic of internal combustion engine dynamics where forces are dominated by gas pressure during combustion and inertial forces during the cycle.

## Input Parameters Used

### Given Constants
- Big end mass (m_big): 3.534 kg
- Big end acceleration (a_big): 2229.67 m/s²

### Gas Force Data (Fg)
Applied at various crank angles representing combustion and expansion forces.

## Formulas Applied

1. **Connecting Rod Angle**: φ = arcsin(sin(θ) / 4)
2. **Load Component Q**: Q = -12.760 × [cos(θ) + cos(2θ)/4 + Fg] / cos(φ)
3. **Connecting Rod Force**: Fcr = m_big × a_big
4. **Resultant Crank Pin Load**: F_PL = √[Q² + Fcr² - 2QFcr×cos(θ + φ)]

## Dependencies
- GNU Octave 8.4.0+ or MATLAB
- No additional toolboxes required

## Author
Generated as part of mechanical engineering analysis for the maneet robotics repository.

## Date
January 2026
