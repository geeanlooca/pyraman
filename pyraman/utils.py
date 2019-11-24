import numpy as np

speed_of_light = c0 = 299792458

# Raman gain for silica fibers
silica_gain = 7e-14

# Effective for area for single-mode fibers
effective_area = 80*(1e-6 ** 2)


def wavelength_to_frequency(lambdas):
    return speed_of_light / lambdas


def frequency_to_wavelength(freqs):
    return speed_of_light / freqs


def alpha_to_linear(alpha):
    return alpha * 1e-3 * np.log(10)/10
