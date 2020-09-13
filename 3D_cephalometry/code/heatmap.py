import numpy as np
from math import exp, log, sqrt, ceil


def gaussian(array_like_hm, mean, sigma):
    """vector version normal distribution pdf"""
    array_like_hm -= mean
    x_term = array_like_hm[:, 0] ** 2
    y_term = array_like_hm[:, 1] ** 2
    z_term = array_like_hm[:, 2] ** 2
    exp_value = - (x_term + y_term+z_term) / 2 / pow(sigma, 2)
    return np.exp(exp_value)

def draw_heatmap(width, height, depth,x, y,z, sigma, array_like_hm):
    m1 = (x,y,z)
    zz = gaussian(array_like_hm, m1, sigma)
    img = zz.reshape((height, width,depth))
    return img










