import numpy as np

speed_of_light = c0 = 299792458


def wavelength_to_frequency(lambdas):
    return speed_of_light / lambdas


def frequency_to_wavelength(freqs):
    return speed_of_light / freqs


def alpha_to_linear(alpha):
    return alpha * 1e-3 * np.log(10)/10
