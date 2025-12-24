"""
Visualization and Animation for F1 Car CFD Simulation
Creates realistic simulation test animations
"""

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import imageio
import os


class SimulationVisualizer:
    """Creates visualizations and animations of CFD simulation"""
    
    def __init__(self, car_mesh, cfd_sim):
        """
        Initialize visualizer
        
        Parameters:
        -----------
        car_mesh : pyvista.PolyData
            F1 car mesh
        cfd_sim : CFDSimulation
            CFD simulation object
        """
        self.car_mesh = car_mesh
        self.cfd_sim = cfd_sim
        self.plotter = None
        
    def create_static_visualization(self, save_path='simulation_static.png'):
        """Create a static visualization of the simulation"""
        print("Creating static visualization...")
        
        # Create plotter
        pl = pv.Plotter(off_screen=True, window_size=[1920, 1080])
        
        # Add car with smooth shading
        pl.add_mesh(
            self.car_mesh,
            color='red',
            metallic=0.5,
            roughness=0.3,
            show_edges=False
        )
        
        # Add ground plane
        ground = pv.Plane(
            center=[0.5, 0, 0],
            direction=[0, 0, 1],
            i_size=10,
            j_size=10
        )
        pl.add_mesh(ground, color='lightgray', opacity=0.3)
        
        # Compute and add streamlines
        seeds = self.cfd_sim.get_streamlines_seeds(n_seeds=30)
        streamlines = self.cfd_sim.compute_streamlines(seeds)
        
        for i, line in enumerate(streamlines):
            if len(line) > 1:
                spline = pv.Spline(line, 100)
                pl.add_mesh(
                    spline,
                    color='blue',
                    line_width=2,
                    opacity=0.6
                )
        
        # Set camera position for good view
        pl.camera_position = [
            (-3, -4, 3),  # Camera position
            (0.5, 0, 0.3),  # Focal point
            (0, 0, 1)  # View up
        ]
        
        # Add title
        pl.add_text(
            "F1 Car CFD Simulation - Streamlines",
            position='upper_edge',
            font_size=16,
            color='black'
        )
        
        pl.screenshot(save_path)
        pl.close()
        
        print(f"Static visualization saved to {save_path}")
        return save_path
    
    def create_particle_animation(self, output_path='simulation_animation.gif', 
                                  n_particles=100, n_frames=120, fps=30):
        """
        Create particle-based flow animation
        
        Parameters:
        -----------
        output_path : str
            Output file path for animation
        n_particles : int
            Number of flow particles
        n_frames : int
            Number of animation frames
        fps : int
            Frames per second
        """
        print(f"Creating particle animation with {n_frames} frames...")
        
        # Initialize particles upstream
        xmin, xmax = self.cfd_sim.x[0], self.cfd_sim.x[-1]
        ymin, ymax = self.cfd_sim.y[0], self.cfd_sim.y[-1]
        zmin, zmax = self.cfd_sim.z[0], self.cfd_sim.z[-1]
        
        x_start = xmin + 0.1 * (xmax - xmin)
        
        # Initialize particle positions
        particles = []
        for _ in range(n_particles):
            y_pos = ymin + np.random.random() * (ymax - ymin)
            z_pos = zmin + 0.2 * (zmax - zmin) + np.random.random() * 0.6 * (zmax - zmin)
            particles.append([x_start, y_pos, z_pos])
        
        particles = np.array(particles)
        
        # Create interpolators for velocity
        from scipy.interpolate import RegularGridInterpolator
        
        u_interp = RegularGridInterpolator(
            (self.cfd_sim.x, self.cfd_sim.y, self.cfd_sim.z),
            self.cfd_sim.velocity_field['u'],
            bounds_error=False,
            fill_value=self.cfd_sim.freestream_velocity
        )
        
        v_interp = RegularGridInterpolator(
            (self.cfd_sim.x, self.cfd_sim.y, self.cfd_sim.z),
            self.cfd_sim.velocity_field['v'],
            bounds_error=False,
            fill_value=0.0
        )
        
        w_interp = RegularGridInterpolator(
            (self.cfd_sim.x, self.cfd_sim.y, self.cfd_sim.z),
            self.cfd_sim.velocity_field['w'],
            bounds_error=False,
            fill_value=0.0
        )
        
        # Store frames
        frames = []
        dt = 0.02
        
        for frame_idx in range(n_frames):
            if frame_idx % 10 == 0:
                print(f"  Rendering frame {frame_idx + 1}/{n_frames}...")
            
            # Create plotter for this frame
            pl = pv.Plotter(off_screen=True, window_size=[1280, 720])
            
            # Add car
            pl.add_mesh(
                self.car_mesh,
                color='red',
                metallic=0.5,
                roughness=0.3
            )
            
            # Add ground plane
            ground = pv.Plane(
                center=[0.5, 0, 0],
                direction=[0, 0, 1],
                i_size=10,
                j_size=10
            )
            pl.add_mesh(ground, color='lightgray', opacity=0.2)
            
            # Update particle positions
            new_particles = []
            for particle in particles:
                # Get velocity at particle position
                vel = np.array([
                    u_interp(particle)[0],
                    v_interp(particle)[0],
                    w_interp(particle)[0]
                ])
                
                # Update position
                new_pos = particle + dt * vel
                
                # Reset particles that exit domain
                if (new_pos[0] > xmax or new_pos[0] < xmin or
                    new_pos[1] > ymax or new_pos[1] < ymin or
                    new_pos[2] > zmax or new_pos[2] < zmin):
                    # Respawn at inlet
                    new_pos = np.array([
                        x_start,
                        ymin + np.random.random() * (ymax - ymin),
                        zmin + 0.2 * (zmax - zmin) + np.random.random() * 0.6 * (zmax - zmin)
                    ])
                
                new_particles.append(new_pos)
            
            particles = np.array(new_particles)
            
            # Visualize particles as small spheres
            for particle in particles:
                sphere = pv.Sphere(radius=0.03, center=particle)
                # Color by velocity
                vel_mag = np.linalg.norm([
                    u_interp(particle)[0],
                    v_interp(particle)[0],
                    w_interp(particle)[0]
                ])
                # Normalize velocity for coloring
                color_val = min(vel_mag / (self.cfd_sim.freestream_velocity * 1.5), 1.0)
                color = plt.cm.jet(color_val)[:3]
                
                pl.add_mesh(sphere, color=color, opacity=0.8)
            
            # Set camera
            pl.camera_position = [
                (-3, -4, 3),
                (0.5, 0, 0.3),
                (0, 0, 1)
            ]
            
            # Add info text
            pl.add_text(
                f"F1 CFD Simulation - Flow Visualization\nFrame: {frame_idx + 1}/{n_frames}",
                position='upper_left',
                font_size=10,
                color='white'
            )
            
            # Add velocity info
            info_text = (f"Freestream Velocity: {self.cfd_sim.freestream_velocity:.1f} m/s\n"
                        f"Particles: {n_particles}")
            pl.add_text(
                info_text,
                position='lower_left',
                font_size=9,
                color='white'
            )
            
            # Capture frame
            frame = pl.screenshot(return_img=True)
            frames.append(frame)
            pl.close()
        
        # Save as GIF
        print(f"Saving animation to {output_path}...")
        imageio.mimsave(output_path, frames, fps=fps, loop=0)
        print(f"Animation saved successfully!")
        
        return output_path
    
    def create_pressure_visualization(self, save_path='pressure_field.png'):
        """Create pressure field visualization"""
        print("Creating pressure field visualization...")
        
        # Create a slice through the car
        cfd = self.cfd_sim
        
        # Create a plane at y=0 (centerline)
        y_idx = len(cfd.y) // 2
        
        # Extract pressure at this slice
        pressure_slice = cfd.pressure_field[:, y_idx, :]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Create meshgrid for plotting
        X_slice = cfd.X[:, y_idx, :]
        Z_slice = cfd.Z[:, y_idx, :]
        
        # Normalize pressure for visualization
        p_min, p_max = pressure_slice.min(), pressure_slice.max()
        
        # Contour plot
        levels = np.linspace(p_min, p_max, 30)
        cf = ax.contourf(X_slice, Z_slice, pressure_slice, levels=levels, cmap='RdYlBu_r')
        
        # Overlay car silhouette
        car_bounds = self.car_mesh.bounds
        car_rect = plt.Rectangle(
            (car_bounds[0], car_bounds[4]),
            car_bounds[1] - car_bounds[0],
            car_bounds[5] - car_bounds[4],
            fill=True,
            facecolor='black',
            edgecolor='white',
            linewidth=2,
            alpha=0.7
        )
        ax.add_patch(car_rect)
        
        # Colorbar
        cbar = plt.colorbar(cf, ax=ax)
        cbar.set_label('Pressure (Pa)', fontsize=12)
        
        # Labels and title
        ax.set_xlabel('X Position (m)', fontsize=12)
        ax.set_ylabel('Z Position (m)', fontsize=12)
        ax.set_title('Pressure Field - Centerline Slice (Y=0)', fontsize=14, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Pressure visualization saved to {save_path}")
        return save_path
    
    def create_velocity_magnitude_plot(self, save_path='velocity_field.png'):
        """Create velocity magnitude visualization"""
        print("Creating velocity field visualization...")
        
        cfd = self.cfd_sim
        
        # Create a slice through the car at centerline
        y_idx = len(cfd.y) // 2
        
        # Compute velocity magnitude
        u_slice = cfd.velocity_field['u'][:, y_idx, :]
        v_slice = cfd.velocity_field['v'][:, y_idx, :]
        w_slice = cfd.velocity_field['w'][:, y_idx, :]
        
        vel_mag = np.sqrt(u_slice**2 + w_slice**2)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))
        
        X_slice = cfd.X[:, y_idx, :]
        Z_slice = cfd.Z[:, y_idx, :]
        
        # Contour plot
        levels = np.linspace(0, vel_mag.max(), 30)
        cf = ax.contourf(X_slice, Z_slice, vel_mag, levels=levels, cmap='viridis')
        
        # Add velocity vectors (quiver plot, subsampled)
        skip = 3
        ax.quiver(
            X_slice[::skip, ::skip],
            Z_slice[::skip, ::skip],
            u_slice[::skip, ::skip],
            w_slice[::skip, ::skip],
            alpha=0.5,
            color='white',
            scale=200
        )
        
        # Overlay car silhouette
        car_bounds = self.car_mesh.bounds
        car_rect = plt.Rectangle(
            (car_bounds[0], car_bounds[4]),
            car_bounds[1] - car_bounds[0],
            car_bounds[5] - car_bounds[4],
            fill=True,
            facecolor='black',
            edgecolor='white',
            linewidth=2,
            alpha=0.7
        )
        ax.add_patch(car_rect)
        
        # Colorbar
        cbar = plt.colorbar(cf, ax=ax)
        cbar.set_label('Velocity Magnitude (m/s)', fontsize=12)
        
        # Labels and title
        ax.set_xlabel('X Position (m)', fontsize=12)
        ax.set_ylabel('Z Position (m)', fontsize=12)
        ax.set_title('Velocity Field - Centerline Slice (Y=0)', fontsize=14, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Velocity visualization saved to {save_path}")
        return save_path
    
    def create_summary_report(self, forces, save_path='simulation_report.txt'):
        """Create a text report of simulation results"""
        print("Creating simulation report...")
        
        report = []
        report.append("=" * 60)
        report.append("F1 CAR CFD SIMULATION REPORT")
        report.append("=" * 60)
        report.append("")
        report.append("SIMULATION PARAMETERS:")
        report.append(f"  Freestream Velocity: {self.cfd_sim.freestream_velocity:.2f} m/s "
                     f"({self.cfd_sim.freestream_velocity * 3.6:.1f} km/h)")
        report.append(f"  Air Density: {self.cfd_sim.air_density:.3f} kg/m³")
        report.append(f"  Dynamic Viscosity: {self.cfd_sim.dynamic_viscosity:.2e} Pa·s")
        report.append(f"  Grid Resolution: {self.cfd_sim.grid_resolution}³ points")
        report.append("")
        report.append("CAR GEOMETRY:")
        bounds = self.car_mesh.bounds
        report.append(f"  Length: {bounds[1] - bounds[0]:.3f} m")
        report.append(f"  Width: {bounds[3] - bounds[2]:.3f} m")
        report.append(f"  Height: {bounds[5] - bounds[4]:.3f} m")
        report.append("")
        report.append("AERODYNAMIC FORCES:")
        report.append(f"  Drag Force: {forces['drag']:.2f} N")
        report.append(f"  Downforce: {forces['downforce']:.2f} N")
        report.append(f"  Drag Coefficient (Cd): {forces['Cd']:.2f}")
        report.append(f"  Lift Coefficient (Cl): {forces['Cl']:.2f}")
        report.append(f"  Dynamic Pressure: {forces['dynamic_pressure']:.2f} Pa")
        report.append("")
        report.append("=" * 60)
        report.append("Note: This is a simplified CFD simulation for visualization")
        report.append("Real F1 aerodynamics require high-fidelity CFD tools")
        report.append("=" * 60)
        
        report_text = "\n".join(report)
        
        with open(save_path, 'w') as f:
            f.write(report_text)
        
        print(report_text)
        print(f"\nReport saved to {save_path}")
        
        return save_path
