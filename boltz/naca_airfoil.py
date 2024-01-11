import torch

def y_thickness(x, t):
    return (t/0.2) * (
        + 0.2969 * torch.sqrt(x)
        - 0.1260 * x
        - 0.3516 * x **2.
        + 0.2843 * x **3.
        - 0.1015 * x **4.
    )

def camber_line(x, p:float, m:float):
    assert x.min() >= 0.0
    y_lower_p = m/p**2 * (2* p * x - x**2.)
    y_lower_p_grad = 2 * m / p**2. * (p - x)

    y_upper_p = m/(1 - p)**2. * (1 - 2 * p + 2 * p * x - x**2.)
    y_upper_p_grad = 2 * m / (1 - p**2.) * (p - x)

    y_c = torch.where(x < p, y_lower_p, y_upper_p)
    y_c_grad = torch.where(x < p, y_lower_p_grad, y_upper_p_grad)
    return y_c, y_c_grad

def upper(x, y_camber, y_thickness, y_camber_grad):
    angle_tensor = torch.arctan(y_camber_grad)
    x_u = x - y_thickness * torch.sin(angle_tensor)
    y_u = y_camber + y_thickness * torch.cos(angle_tensor)
    return x_u, y_u

def lower(x, y_camber, y_thickness, y_camber_grad):
    angle_tensor = torch.arctan(y_camber_grad)
    x_u = x + y_thickness * torch.sin(angle_tensor)
    y_u = y_camber - y_thickness * torch.cos(angle_tensor)
    return x_u, y_u

def airfoil(size: int, m: float, p: float, thickness: float):
    # Change scale to match NACA's
    M = m / 100
    P = p / 10
    T = thickness / 100

    x = torch.arange(0, 1, 1/size)
    y_t = y_thickness(x, T)
    y_c, y_c_grad = camber_line(x, P, M)
    _, y_upper = upper(x, y_c, y_t, y_c_grad)
    _, y_lower = lower(x, y_c, y_t, y_c_grad)

    #Shift everything to positive y
    y_min = torch.abs(torch.min(y_lower))
    y_lower += y_min
    y_upper += y_min

    #Scale shape
    y_lower *= size
    y_upper *= size
    y_max = torch.max(y_upper).ceil().int().item()

    xg, yg = torch.meshgrid(
        torch.arange(size),
        torch.arange(y_max + 1),
    )
    return torch.logical_and(
        torch.le(yg.T, y_upper),
        torch.ge(yg.T, y_lower)
        )
