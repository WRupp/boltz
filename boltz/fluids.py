from dataclasses import dataclass

@dataclass
class Fluid:
    sound_speed: float
    kinematic_viscosity: float

class LatticeFluid:

    def __init__(self, fluid:Fluid, delta_time: int) -> None:
        self.relaxation_time = (
            fluid.kinematic_viscosity / fluid.sound_speed**2.0 
            + delta_time / 2.0
        )
    
    @property
    def tau(self):
        """Alias for relaxation_time"""
        return self.relaxation_time