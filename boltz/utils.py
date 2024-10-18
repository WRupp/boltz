import torch

def sin_source(
        lattice,
        mask: torch.Tensor,
        anchor: tuple[int, int],
        time: int,
        amplitude: float = 0.1,
        frequency: int = 45,
    ):
    """A sinusoidal vibration source boundary"""
    s = mask.shape
    local_moments = lattice.moments[:, anchor[0]:anchor[0]+s[0], anchor[1]:anchor[1]+s[1]]

    noise = amplitude * torch.sin(torch.tensor([ time / frequency * 2.0 * torch.pi]))
    changed_moments = local_moments + torch.ones(s) * noise

    expanded_mask = mask.unsqueeze(0).expand((lattice.q_dims, -1, -1))
    lattice.moments[
        :, 
        anchor[0]:anchor[0]+s[0], 
        anchor[1]:anchor[1]+s[1]
    ] = torch.where(expanded_mask, changed_moments, local_moments)

def bounce_back(lattice, mask: torch.Tensor, anchor: tuple[int,int]):
    """An all-directions bounce-back boundary"""
    s = mask.shape
    local_moments = lattice.moments[:, anchor[0]:anchor[0]+s[0], anchor[1]:anchor[1]+s[1]]
    
    changed_moments = torch.flip(local_moments, [0, 1])

    expanded_mask = mask.unsqueeze(0).expand((lattice.q_dims, -1, -1))
    lattice.moments[
        :, 
        anchor[0]:anchor[0]+s[0], 
        anchor[1]:anchor[1]+s[1]
    ] = torch.where(expanded_mask, changed_moments, local_moments)

def accelerate(lattice, mask: torch.Tensor, anchor: tuple[int,int], force_vector: list[float]):
    s = mask.shape
    local_moments = lattice.moments[:, anchor[0]:anchor[0]+s[0], anchor[1]:anchor[1]+s[1]]
    
    moments_forced = torch.zeros(lattice.q_dims, 1)
    for force, versor in zip(force_vector, lattice.versors):
        moments_forced += force * versor / (2 * lattice.sound_speed_squared)
    changed_moments = local_moments + moments_forced.unsqueeze(dim=1).expand(lattice.q_dims, s[0], s[1])    

    expanded_mask = mask.unsqueeze(0).expand((lattice.q_dims, -1, -1))
    lattice.moments[
        :, 
        anchor[0]:anchor[0]+s[0], 
        anchor[1]:anchor[1]+s[1]
    ] = torch.where(expanded_mask, changed_moments, local_moments)

