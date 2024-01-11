from collections import namedtuple

import torch

from .fluids import LatticeFluid

Directions = namedtuple('Directions', ["x", "y"])

class D2Q9:
    d_dims = 2
    q_dims = 9
    sound_speed_squared = 1.0/3.0

    weights = torch.tensor([
        [1./36, 1./9, 1./36],
        [1./9,  4/9,  1./9],
        [1./36, 1./9, 1./36],
    ]).view((9,1))

    versors = Directions(
        x=torch.tensor([
            [-1., -1., -1.],
            [-0., -0., -0.],
            [+1., +1., +1.],
        ]).view((9,1)),
        y=torch.tensor([
            [+1., +0., -1.],
            [+1., +0., -1.],
            [+1., +0., -1.],
        ]).view((9,1))
    )

class Lattice(D2Q9):

    def __init__(self, fluid, delta_time, delta_space, shape):
        self.delta_time = delta_time
        self.delta_space = delta_space
        self.fluid = LatticeFluid(
            fluid=fluid, 
            delta_time=self.delta_time, 
            delta_space=self.delta_space
            )

        self.shape = shape
        self._init_moments()

    def _init_moments(self):
        """Initiate moments at equilibrium"""
        self._moments = (
            torch.ones(self.q_dims, *self.shape) * self.weights.unsqueeze(2)
        ).view((self.q_dims, -1))

    @property
    def moments(self):
        return self._moments.view(-1, *self.shape)

    def density(self):
        return self._moments.sum(dim=0)

    #TODO: Remove 2D dependence. Vectorize as an N-dim Op.
    def velocity(self, density):
        return Directions(
            x = 1./density * (self._moments * self.versors.x).sum(dim=0),
            y = 1./density * (self._moments * self.versors.y).sum(dim=0),
        )

    #TODO: Remove 2D dependence. Vectorize as an N-dim Op.
    def equilibrium(self, density, velocity):
        cs2 = self.sound_speed_squared
        v_u = self.versors.x * velocity.x + self.versors.y * velocity.y
        u_u = velocity.x**2.0 + velocity.y**2.0
        eq_expression = ( 
            1 
            + (1./cs2) * v_u 
            + 0.5 * (1./cs2)**2.0 * v_u**2.0 
            - 0.5 * (1./cs2) * u_u
            )
        return self.weights * density * eq_expression

    def relaxation(self, equilibrium):
        return self._moments - 1./self.fluid.tau * (self._moments - equilibrium)

    def collide(self):
        density = self.density()
        velocity = self.velocity(density)
        equilibrium = self.equilibrium(density, velocity)
        self._moments = self.relaxation(equilibrium)

        return density, velocity, equilibrium

    #TODO: Remove 2D dependence. Vectorize as an N-dim Op.
    def stream(self):
        for idx, direction in enumerate(self.versors):
            for c, step in enumerate(direction):
                channel = self.moments[c, :, :]
                self.moments[c, :, :] = torch.roll(
                    channel,
                    shifts=int(step.item()),
                    dims=idx,
                )