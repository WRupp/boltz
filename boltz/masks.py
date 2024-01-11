import torch

from .naca_airfoil import airfoil

def point():
    """Returns a single point mask."""
    return rectangle(height=1, width=1)

def rectangle(height: int, width: int):
    """Returns a rectangular shaped mask"""
    return torch.ones((height, width)).bool()

def circle(radius: int):
    """Return a mask tensor of a circle. The lenght of the each dim is 2*radius+1"""
    diameter = radius * 2
    x_indices, y_indices = torch.meshgrid(
        torch.arange(diameter+1),
        torch.arange(diameter+1),
    )
    center = torch.tensor([radius, radius])
    
    distances = torch.sqrt((x_indices - center[0])**2.0 + (y_indices - center[1])**2.0 )
    return distances <= radius

def wing(size: int, m:float, p:float, thickness:float):
    """Returns a mask tensor with a wing-like pattern"""
    return airfoil(size=size, m=m, p=p, thickness=thickness)