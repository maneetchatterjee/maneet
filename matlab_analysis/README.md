# Crank Pin Load Analysis

This directory contains the MATLAB/Octave analysis for calculating and visualizing the resultant crank pin load (F_PL) as a function of crank angle.

## Files

- `crank_pin_load.m` - Main MATLAB/Octave script that performs the calculations
- `fpl_plot.png` - Generated plot showing F_PL vs Crank Angle

## Analysis Overview

### Input Parameters
- **Crank Angles (θ)**: 0° to 720° in increments of 20° to 30°
- **Gas Force (Fg)**: Variable force values ranging from -0.10 kN to 85 kN
- **Big End Mass (m_big)**: 3.534 kg
- **Big End Acceleration (a_big)**: 2229.67 m/s²

### Calculations Performed
1. Convert crank angles to radians
2. Calculate connecting rod angle (φ) using: φ = arcsin(sin(θ)/4)
3. Calculate force Q using the formula:
   ```
   Q = 1278.79 × (cos(θ) + cos(2θ)/4 + Fg) / cos(φ)
   ```
4. Calculate connecting rod force: Fcr = m_big × a_big
5. Calculate resultant crank pin load:
   ```
   F_PL = √(Q² + Fcr² - 2·Q·Fcr·cos(θ + φ))
   ```

### Results

**Average Resultant Crank Pin Load: 17,279.99 kN**

The analysis shows that:
- Maximum load occurs at 20° crank angle: 109,091.00 kN
- Minimum load occurs at 570° crank angle: 7.26 kN
- The load pattern is cyclic, completing two cycles over 720° (two complete revolutions)

### Running the Analysis

To run the analysis:

```bash
cd matlab_analysis
octave --no-gui --eval "run('crank_pin_load.m')"
```

Or in MATLAB:
```matlab
cd matlab_analysis
run('crank_pin_load.m')
```

## Output

The script produces:
1. A table of crank angles and corresponding F_PL values
2. A plot (saved as `fpl_plot.png`) showing the relationship between crank angle and resultant load
3. The calculated average resultant crank pin load

## Dependencies

- MATLAB R2016b or later, OR
- GNU Octave 4.0 or later (open-source alternative)
