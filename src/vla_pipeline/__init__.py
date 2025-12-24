"""
VLA Pipeline - Vision-Language-Action Pipeline for Robotic Manipulation

A modular pipeline for robotic manipulation in simulation that integrates:
- Computer vision-based perception
- Natural language understanding
- Symbolic action planning
- Inverse kinematics control
- Physics simulation (PyBullet)
"""

__version__ = "0.1.0"
__author__ = "Maneet"

from .pipeline import VLAPipeline

__all__ = ["VLAPipeline"]
