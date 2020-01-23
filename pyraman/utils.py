from enum import Enum

import numpy as np

import scipy.constants

from scipy.constants import speed_of_light as c0


def wavelength_to_frequency(lambdas):
    """Convert wavelength to frequency."""
    return scipy.constants.lambda2nu(lambdas)


def frequency_to_wavelength(freqs):
    """Convert frequency to wavelength."""
    return scipy.constants.nu2lambda(freqs)


def alpha_to_linear(alpha):
    """Convert attenuation constant from dB to linear units."""
    return alpha * 1e-3 * np.log(10)/10


def alpha2linear(alpha):
    return alpha_to_linear(alpha)

def wavelength2frequency(lambdas):
    return wavelength_to_frequency(lambdas)

def frequency2wavelength(freqs):
    return frequency_to_wavelength(freqs)

def watt_to_dBm(power):
    return 10 * np.log10(power) + 30

def dBm_to_watt(power):
    return 10**((power-30)/10)

def watt2dBm(power):
    return watt_to_dBm(power)

def dBm2watt(power):
    return dBm_to_watt(power)


def wdm_comb(num, center_wavelength, spacing):
    center_freq = wavelength2frequency(center_wavelength)
    if num % 2:
        freqs = np.arange(-(num-1)/2, (num-1)/2 + 1)
    else:
        freqs = np.arange(- num/2) - (num+1)/2
        freqs = np.arange(-(num-1)/2, (num-1)/2 + 1)
        
    comb = center_freq + spacing * freqs
    
    return np.sort(frequency2wavelength(comb))
        

class bands(Enum):
    """Class enumerating the optical transmission bandwidths.

    Data taken from: https://www.thefoa.org/tech/ref/basic/SMbands.html
    """
    O = (1260e-9, 1360e-9)
    E = (1360e-9, 1460e-9)
    S = (1460e-9, 1530e-9)
    C = (1530e-9, 1565e-9)
    L = (1565e-9, 1625e-9)
    U = (1625e-9, 1675e-9)
    
    def __add__(self, other):
        joint = self.value + other.value        
        return (min(joint), max(joint))
    
    
        


"""
O-band 	1260 – 1360 nm 	Original band, PON upstream
E-band 	1360 – 1460 nm 	Water peak band
S-band 	1460 – 1530 nm 	PON downstream
C-band 	1530 – 1565 nm 	Lowest attenuation, original DWDM band, compatible with fiber amplifiers, CATV
L-band 	1565 – 1625 nm 	Low attenuation, expanded DWDM band
U-band 	1625 – 1675 nm 	Ultra-long wavelength
"""