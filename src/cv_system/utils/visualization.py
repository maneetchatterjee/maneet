"""
Visualization Utilities
Helper functions for visualizing CV results
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class Visualizer:
    """Unified visualization interface"""
    
    def __init__(self, window_name: str = "Computer Vision System"):
        """
        Initialize visualizer
        
        Args:
            window_name: Name of display window
        """
        self.window_name = window_name
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.thickness = 2
        
    def create_grid(self, images: List[np.ndarray], 
                    labels: Optional[List[str]] = None,
                    grid_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Create grid of images
        
        Args:
            images: List of images
            labels: Optional labels for each image
            grid_size: Optional (rows, cols), auto-calculated if None
            
        Returns:
            Grid image
        """
        if not images:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Calculate grid size
        if grid_size is None:
            n = len(images)
            cols = int(np.ceil(np.sqrt(n)))
            rows = int(np.ceil(n / cols))
        else:
            rows, cols = grid_size
        
        # Resize all images to same size
        target_h = images[0].shape[0]
        target_w = images[0].shape[1]
        
        resized_images = []
        for img in images:
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            resized = cv2.resize(img, (target_w, target_h))
            
            # Add label if provided
            if labels and len(resized_images) < len(labels):
                label = labels[len(resized_images)]
                cv2.putText(resized, label, (10, 30),
                           self.font, self.font_scale, (255, 255, 255), self.thickness)
            
            resized_images.append(resized)
        
        # Pad with blank images if needed
        while len(resized_images) < rows * cols:
            resized_images.append(np.zeros((target_h, target_w, 3), dtype=np.uint8))
        
        # Create grid
        grid_rows = []
        for i in range(rows):
            row_images = resized_images[i*cols:(i+1)*cols]
            grid_row = np.hstack(row_images)
            grid_rows.append(grid_row)
        
        grid = np.vstack(grid_rows)
        return grid
    
    def draw_info_panel(self, frame: np.ndarray, info: Dict) -> np.ndarray:
        """
        Draw information panel on frame
        
        Args:
            frame: Input frame
            info: Dictionary of information to display
            
        Returns:
            Frame with info panel
        """
        vis_frame = frame.copy()
        h, w = vis_frame.shape[:2]
        
        # Create semi-transparent overlay
        overlay = vis_frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 30 + len(info) * 25), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, vis_frame, 0.3, 0, vis_frame)
        
        # Draw text
        y_offset = 30
        for key, value in info.items():
            text = f"{key}: {value}"
            cv2.putText(vis_frame, text, (20, y_offset),
                       self.font, 0.5, (255, 255, 255), 1)
            y_offset += 25
        
        return vis_frame
    
    def draw_fps(self, frame: np.ndarray, fps: float) -> np.ndarray:
        """
        Draw FPS counter
        
        Args:
            frame: Input frame
            fps: FPS value
            
        Returns:
            Frame with FPS
        """
        text = f"FPS: {fps:.1f}"
        cv2.putText(frame, text, (frame.shape[1] - 120, 30),
                   self.font, 0.7, (0, 255, 0), 2)
        return frame
    
    def draw_heatmap(self, frame: np.ndarray, heatmap: np.ndarray, 
                     alpha: float = 0.5) -> np.ndarray:
        """
        Overlay heatmap on frame
        
        Args:
            frame: Original frame
            heatmap: Heatmap (normalized 0-1)
            alpha: Overlay transparency
            
        Returns:
            Frame with heatmap overlay
        """
        # Normalize heatmap
        heatmap_norm = (heatmap * 255).astype(np.uint8)
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
        
        # Resize to match frame
        if heatmap_colored.shape[:2] != frame.shape[:2]:
            heatmap_colored = cv2.resize(heatmap_colored, (frame.shape[1], frame.shape[0]))
        
        # Blend
        result = cv2.addWeighted(frame, 1 - alpha, heatmap_colored, alpha, 0)
        return result
    
    def create_side_by_side(self, left: np.ndarray, right: np.ndarray,
                           left_label: str = "", right_label: str = "") -> np.ndarray:
        """
        Create side-by-side comparison
        
        Args:
            left: Left image
            right: Right image
            left_label: Label for left image
            right_label: Label for right image
            
        Returns:
            Combined image
        """
        # Ensure same height
        h = max(left.shape[0], right.shape[0])
        
        if left.shape[0] != h:
            left = cv2.resize(left, (int(left.shape[1] * h / left.shape[0]), h))
        if right.shape[0] != h:
            right = cv2.resize(right, (int(right.shape[1] * h / right.shape[0]), h))
        
        # Ensure 3 channels
        if len(left.shape) == 2:
            left = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
        if len(right.shape) == 2:
            right = cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)
        
        # Add labels
        if left_label:
            cv2.putText(left, left_label, (10, 30),
                       self.font, self.font_scale, (255, 255, 255), self.thickness)
        if right_label:
            cv2.putText(right, right_label, (10, 30),
                       self.font, self.font_scale, (255, 255, 255), self.thickness)
        
        # Concatenate
        combined = np.hstack([left, right])
        return combined
    
    def show(self, frame: np.ndarray):
        """Display frame"""
        cv2.imshow(self.window_name, frame)
    
    def close(self):
        """Close all windows"""
        cv2.destroyAllWindows()


def draw_3d_bbox(frame: np.ndarray, corners: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0)):
    """
    Draw 3D bounding box
    
    Args:
        frame: Input frame
        corners: 8x2 array of 2D projected corners
        color: Box color
        
    Returns:
        Frame with 3D box
    """
    vis_frame = frame.copy()
    
    # Draw edges of the box
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Front face
        (4, 5), (5, 6), (6, 7), (7, 4),  # Back face
        (0, 4), (1, 5), (2, 6), (3, 7)   # Connecting edges
    ]
    
    for edge in edges:
        pt1 = tuple(map(int, corners[edge[0]]))
        pt2 = tuple(map(int, corners[edge[1]]))
        cv2.line(vis_frame, pt1, pt2, color, 2)
    
    return vis_frame


def create_attention_map(frame: np.ndarray, attention_weights: np.ndarray) -> np.ndarray:
    """
    Create attention map visualization
    
    Args:
        frame: Original frame
        attention_weights: Attention weights
        
    Returns:
        Attention visualization
    """
    # Resize attention to frame size
    attention_resized = cv2.resize(attention_weights, (frame.shape[1], frame.shape[0]))
    
    # Normalize
    attention_norm = (attention_resized - attention_resized.min()) / (attention_resized.max() - attention_resized.min())
    
    # Apply colormap
    attention_colored = cv2.applyColorMap((attention_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    # Blend with original
    result = cv2.addWeighted(frame, 0.6, attention_colored, 0.4, 0)
    
    return result
