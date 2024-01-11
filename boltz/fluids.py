from dataclasses import dataclass

@dataclass
class Fluid:
    sound_speed: float
    kinematic_viscosity: float

@dataclass
class LatticeFluid:
    fluid: Fluid
    delta_time: int
    delta_space: int

    @property
    def relaxation_time(self):
        return (
            self.fluid.kinematic_viscosity / self.fluid.sound_speed**2.0 
            + self.delta_time / 2.0
        )
    
    @property
    def tau(self):
        """Alias for relaxation_time"""
        return self.relaxation_time