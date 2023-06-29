import torch
import math
from scipy.integrate import quad


def derivative(y, t, device) : 
    return torch.autograd.grad(
        y, t, create_graph=True,
        grad_outputs=torch.ones(y.size()).to(device))[0]


'''
    making relaxation of Dirac function
'''
def gaussian(x, mu, sigma):

    coefficient = 1.0 / (2 * math.pi * sigma)
    exponent = -(x - mu) ** 2 / (2 * sigma**2)
    return coefficient * torch.exp(exponent)

def bump(x,a=1.0):
    if type(x) == torch.Tensor:
        condition = (x.abs() < a)
        valid_values = torch.exp(x**2 / a**2*(a**2 - x**2)) 
        result = torch.where(condition, valid_values, torch.zeros_like(x))
        return result
    else:
        if abs(x) < a: return math.exp(x**2/a**2* (a**2 - x**2)) 
        else: return 0.0


def normalized_bump(x, a = 1.0):

    integral, _ = quad(lambda x: bump(x,a), -a, a)
    coefficient = 1 / integral

    return coefficient * bump(x,a)


# Figure

