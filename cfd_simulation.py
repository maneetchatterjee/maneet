"""
Simplified CFD Simulation for F1 Car
Implements potential flow simulation around the car geometry
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator


class CFDSimulation:
    """Computational Fluid Dynamics simulation for airflow around F1 car"""
    
    def __init__(self, car_mesh, freestream_velocity=30.0):
        """
        Initialize CFD simulation
        
        Parameters:
        -----------
        car_mesh : pyvista.PolyData
            The F1 car mesh
        freestream_velocity : float
            Freestream velocity in m/s (default: 30 m/s ~ 108 km/h)
        """
        self.car_mesh = car_mesh
        self.freestream_velocity = freestream_velocity
        self.bounds = car_mesh.bounds
        
        # Flow properties
        self.air_density = 1.225  # kg/m³ at sea level
        self.dynamic_viscosity = 1.81e-5  # Pa·s
        
        # Grid for flow field
        self.grid_resolution = 50
        self.flow_field = None
        self.velocity_field = None
        self.pressure_field = None
        
    def setup_domain(self, padding=2.0):
        """Create computational domain around the car"""
        xmin, xmax, ymin, ymax, zmin, zmax = self.bounds
        
        # Extend domain beyond car
        domain_bounds = [
            xmin - padding, xmax + padding,
            ymin - padding, ymax + padding,
            zmin - 0.5, zmax + padding
        ]
        
        # Create grid points
        self.x = np.linspace(domain_bounds[0], domain_bounds[1], self.grid_resolution)
        self.y = np.linspace(domain_bounds[2], domain_bounds[3], self.grid_resolution)
        self.z = np.linspace(domain_bounds[4], domain_bounds[5], self.grid_resolution)
        
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')
        
        return domain_bounds
    
    def compute_flow_field(self):
        """
        Compute simplified flow field using potential flow with rankine body approach
        """
        print("Computing flow field...")
        
        # Initialize velocity components
        u = np.ones_like(self.X) * self.freestream_velocity  # x-direction (streamwise)
        v = np.zeros_like(self.Y)  # y-direction
        w = np.zeros_like(self.Z)  # z-direction
        
        # Get car center and characteristic length
        car_center = np.array([
            (self.bounds[0] + self.bounds[1]) / 2,
            (self.bounds[2] + self.bounds[3]) / 2,
            (self.bounds[4] + self.bounds[5]) / 2
        ])
        
        car_length = self.bounds[1] - self.bounds[0]
        car_width = self.bounds[3] - self.bounds[2]
        car_height = self.bounds[5] - self.bounds[4]
        
        # Simplified flow disturbance around car (dipole + source/sink)
        for i in range(self.grid_resolution):
            for j in range(self.grid_resolution):
                for k in range(self.grid_resolution):
                    point = np.array([self.X[i, j, k], self.Y[i, j, k], self.Z[i, j, k]])
                    r_vec = point - car_center
                    r = np.linalg.norm(r_vec)
                    
                    # Check if point is near the car
                    if r < 0.1:
                        continue
                    
                    # Simplified potential flow around bluff body
                    # Add disturbance that decays with distance
                    
                    # Characteristic radius for the car (ellipsoid approximation)
                    r_char = np.sqrt(
                        (r_vec[0] / (car_length / 2))**2 +
                        (r_vec[1] / (car_width / 2))**2 +
                        (r_vec[2] / (car_height / 2))**2
                    )
                    
                    if r_char < 1.5:  # Near the car
                        # Flow acceleration over the top (Bernoulli effect)
                        if point[2] > car_center[2]:
                            height_factor = (point[2] - car_center[2]) / car_height
                            u[i, j, k] += self.freestream_velocity * 0.5 * height_factor / (r_char**2)
                        
                        # Flow around sides
                        lateral_dist = abs(r_vec[1])
                        if lateral_dist > car_width / 3:
                            v[i, j, k] += np.sign(r_vec[1]) * self.freestream_velocity * 0.3 / (r_char**2)
                        
                        # Downwash behind rear wing
                        if r_vec[0] > car_length * 0.3 and point[2] > car_center[2]:
                            w[i, j, k] -= self.freestream_velocity * 0.2 / (r_char**2)
                        
                        # Wake behind the car
                        if r_vec[0] > 0:
                            wake_factor = np.exp(-lateral_dist / car_width)
                            u[i, j, k] *= (1.0 - 0.4 * wake_factor / (1 + r_vec[0] / car_length))
        
        self.velocity_field = {
            'u': u,
            'v': v,
            'w': w
        }
        
        # Compute pressure field using Bernoulli equation
        velocity_magnitude = np.sqrt(u**2 + v**2 + w**2)
        # P + 0.5 * rho * V^2 = constant
        p_freestream = 101325  # Pa (atmospheric)
        dynamic_pressure_freestream = 0.5 * self.air_density * self.freestream_velocity**2
        
        self.pressure_field = (p_freestream + dynamic_pressure_freestream - 
                              0.5 * self.air_density * velocity_magnitude**2)
        
        print("Flow field computation complete!")
        return self.velocity_field, self.pressure_field
    
    def get_streamlines_seeds(self, n_seeds=20):
        """Generate seed points for streamlines"""
        xmin, xmax = self.x[0], self.x[-1]
        ymin, ymax = self.y[0], self.y[-1]
        zmin, zmax = self.z[0], self.z[-1]
        
        # Create seeds upstream of the car
        x_seed = xmin + 0.1 * (xmax - xmin)
        
        seeds = []
        # Multiple heights
        for z_frac in np.linspace(0.2, 0.8, n_seeds // 2):
            for y_frac in np.linspace(0.3, 0.7, 2):
                y_seed = ymin + y_frac * (ymax - ymin)
                z_seed = zmin + z_frac * (zmax - zmin)
                seeds.append([x_seed, y_seed, z_seed])
        
        return np.array(seeds)
    
    def compute_streamlines(self, seeds, max_steps=500):
        """
        Compute streamlines from seed points
        
        Parameters:
        -----------
        seeds : array
            Starting points for streamlines
        max_steps : int
            Maximum integration steps
        
        Returns:
        --------
        list of streamlines (each is an array of points)
        """
        print("Computing streamlines...")
        
        # Create interpolators for velocity field
        u_interp = RegularGridInterpolator(
            (self.x, self.y, self.z), 
            self.velocity_field['u'],
            bounds_error=False,
            fill_value=self.freestream_velocity
        )
        
        v_interp = RegularGridInterpolator(
            (self.x, self.y, self.z),
            self.velocity_field['v'],
            bounds_error=False,
            fill_value=0.0
        )
        
        w_interp = RegularGridInterpolator(
            (self.x, self.y, self.z),
            self.velocity_field['w'],
            bounds_error=False,
            fill_value=0.0
        )
        
        streamlines = []
        dt = 0.05  # Time step for integration
        
        for seed in seeds:
            line = [seed.copy()]
            point = seed.copy()
            
            for step in range(max_steps):
                # Get velocity at current point
                vel = np.array([
                    u_interp(point)[0],
                    v_interp(point)[0],
                    w_interp(point)[0]
                ])
                
                # Normalize and step
                vel_mag = np.linalg.norm(vel)
                if vel_mag < 0.1:  # Stagnation
                    break
                
                point = point + dt * vel
                
                # Check if out of bounds
                if (point[0] < self.x[0] or point[0] > self.x[-1] or
                    point[1] < self.y[0] or point[1] > self.y[-1] or
                    point[2] < self.z[0] or point[2] > self.z[-1]):
                    break
                
                line.append(point.copy())
            
            if len(line) > 10:  # Only keep substantial streamlines
                streamlines.append(np.array(line))
        
        print(f"Generated {len(streamlines)} streamlines")
        return streamlines
    
    def compute_forces(self):
        """Estimate aerodynamic forces (simplified)"""
        # This is a very simplified force calculation
        # In reality, would need surface integration of pressure and shear stress
        
        # Find car region
        car_center = np.array([
            (self.bounds[0] + self.bounds[1]) / 2,
            (self.bounds[2] + self.bounds[3]) / 2,
            (self.bounds[4] + self.bounds[5]) / 2
        ])
        
        # Reference area (frontal area approximation)
        car_width = self.bounds[3] - self.bounds[2]
        car_height = self.bounds[5] - self.bounds[4]
        A_ref = car_width * car_height
        
        # Simplified drag and downforce coefficients for F1 car
        Cd = 0.7  # Typical F1 drag coefficient
        Cl = -3.0  # Negative lift (downforce)
        
        q = 0.5 * self.air_density * self.freestream_velocity**2
        
        drag = Cd * q * A_ref
        downforce = Cl * q * A_ref
        
        return {
            'drag': drag,
            'downforce': abs(downforce),
            'Cd': Cd,
            'Cl': Cl,
            'dynamic_pressure': q
        }
