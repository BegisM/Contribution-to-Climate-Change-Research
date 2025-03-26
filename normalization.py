import numpy as np


def get_standard_deviation(x):
    mu = get_mu(x)
    sigma = ((x - mu)**2).sum() / x.size
    return np.sqrt(sigma)

def get_mu(x):
    mu = x.sum() / x.size
    return mu


def z_normalization(x):
    mu = get_mu(x)
    sigma = get_standard_deviation(x)
    return ((x - mu) / sigma), mu, sigma


def normalize_back(x, mu, sigma):
    return x * sigma + mu
