"""
3D F1 Car Model Generator
Creates a simplified but recognizable F1 car geometry for CFD simulation
"""

import numpy as np
import pyvista as pv


class F1CarModel:
    """Creates a 3D model of an F1 car"""
    
    def __init__(self, scale=1.0):
        self.scale = scale
        self.car_mesh = None
        
    def create_car(self):
        """Generate the F1 car geometry"""
        # Create main body (monocoque)
        body = self._create_body()
        
        # Create front wing
        front_wing = self._create_front_wing()
        
        # Create rear wing
        rear_wing = self._create_rear_wing()
        
        # Create wheels
        wheels = self._create_wheels()
        
        # Create nose cone
        nose = self._create_nose()
        
        # Create engine cover/airbox
        airbox = self._create_airbox()
        
        # Combine all parts
        self.car_mesh = body + front_wing + rear_wing + wheels + nose + airbox
        
        return self.car_mesh
    
    def _create_body(self):
        """Create main body/monocoque"""
        # Main cockpit section
        body = pv.Box(bounds=[
            -0.5 * self.scale, 1.5 * self.scale,  # x
            -0.4 * self.scale, 0.4 * self.scale,  # y
            0.1 * self.scale, 0.5 * self.scale    # z
        ])
        
        # Sidepods
        left_sidepod = pv.Box(bounds=[
            -0.2 * self.scale, 1.2 * self.scale,  # x
            0.4 * self.scale, 0.7 * self.scale,   # y
            0.1 * self.scale, 0.4 * self.scale    # z
        ])
        
        right_sidepod = pv.Box(bounds=[
            -0.2 * self.scale, 1.2 * self.scale,  # x
            -0.7 * self.scale, -0.4 * self.scale, # y
            0.1 * self.scale, 0.4 * self.scale    # z
        ])
        
        body = body + left_sidepod + right_sidepod
        return body
    
    def _create_front_wing(self):
        """Create front wing assembly"""
        # Main wing element
        wing = pv.Box(bounds=[
            -0.8 * self.scale, -0.6 * self.scale,  # x (front)
            -0.9 * self.scale, 0.9 * self.scale,   # y (wide)
            0.05 * self.scale, 0.15 * self.scale   # z (thin)
        ])
        
        # Second element
        wing2 = pv.Box(bounds=[
            -0.75 * self.scale, -0.65 * self.scale,
            -0.85 * self.scale, 0.85 * self.scale,
            0.15 * self.scale, 0.22 * self.scale
        ])
        
        return wing + wing2
    
    def _create_rear_wing(self):
        """Create rear wing assembly"""
        # Main wing
        wing = pv.Box(bounds=[
            1.8 * self.scale, 2.0 * self.scale,    # x (rear)
            -0.7 * self.scale, 0.7 * self.scale,   # y
            0.6 * self.scale, 0.7 * self.scale     # z (elevated)
        ])
        
        # Upper element (DRS)
        wing2 = pv.Box(bounds=[
            1.85 * self.scale, 2.0 * self.scale,
            -0.65 * self.scale, 0.65 * self.scale,
            0.75 * self.scale, 0.82 * self.scale
        ])
        
        # Wing supports
        support_left = pv.Box(bounds=[
            1.9 * self.scale, 1.95 * self.scale,
            0.5 * self.scale, 0.55 * self.scale,
            0.3 * self.scale, 0.7 * self.scale
        ])
        
        support_right = pv.Box(bounds=[
            1.9 * self.scale, 1.95 * self.scale,
            -0.55 * self.scale, -0.5 * self.scale,
            0.3 * self.scale, 0.7 * self.scale
        ])
        
        return wing + wing2 + support_left + support_right
    
    def _create_wheels(self):
        """Create four wheels"""
        wheel_radius = 0.33 * self.scale
        wheel_thickness = 0.25 * self.scale
        
        # Front left
        fl = pv.Cylinder(
            center=[-0.6 * self.scale, 0.6 * self.scale, 0.33 * self.scale],
            direction=[0, 1, 0],
            radius=wheel_radius,
            height=wheel_thickness
        )
        
        # Front right
        fr = pv.Cylinder(
            center=[-0.6 * self.scale, -0.6 * self.scale, 0.33 * self.scale],
            direction=[0, 1, 0],
            radius=wheel_radius,
            height=wheel_thickness
        )
        
        # Rear left
        rl = pv.Cylinder(
            center=[1.5 * self.scale, 0.6 * self.scale, 0.33 * self.scale],
            direction=[0, 1, 0],
            radius=wheel_radius,
            height=wheel_thickness
        )
        
        # Rear right
        rr = pv.Cylinder(
            center=[1.5 * self.scale, -0.6 * self.scale, 0.33 * self.scale],
            direction=[0, 1, 0],
            radius=wheel_radius,
            height=wheel_thickness
        )
        
        return fl + fr + rl + rr
    
    def _create_nose(self):
        """Create nose cone"""
        # Nose cone (pointed front)
        nose = pv.Cone(
            center=[-0.8 * self.scale, 0, 0.25 * self.scale],
            direction=[1, 0, 0],
            height=0.3 * self.scale,
            radius=0.15 * self.scale
        )
        
        return nose
    
    def _create_airbox(self):
        """Create engine cover and airbox"""
        # Air intake above driver
        airbox = pv.Box(bounds=[
            0.3 * self.scale, 0.7 * self.scale,
            -0.25 * self.scale, 0.25 * self.scale,
            0.5 * self.scale, 0.7 * self.scale
        ])
        
        # Engine cover (tapers to rear)
        engine_cover = pv.Box(bounds=[
            0.7 * self.scale, 1.8 * self.scale,
            -0.35 * self.scale, 0.35 * self.scale,
            0.3 * self.scale, 0.5 * self.scale
        ])
        
        return airbox + engine_cover
    
    def get_mesh(self):
        """Return the car mesh"""
        if self.car_mesh is None:
            self.create_car()
        return self.car_mesh
    
    def get_bounds(self):
        """Get bounding box of the car"""
        if self.car_mesh is None:
            self.create_car()
        return self.car_mesh.bounds
