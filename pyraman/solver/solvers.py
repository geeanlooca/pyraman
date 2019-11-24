import numpy as np
import scipy.integrate
from pyraman.response import gain_spectrum
from pyraman.utils import wavelength_to_frequency, effective_area, silica_gain, alpha_to_linear
import matplotlib.pyplot as plt
from pdb import set_trace


def solve(input_power, input_wavelength, z, losses=0.2, raman_coefficient=silica_gain,
          effective_core_area=effective_area, check_photon_count=False):

    num_signals = input_power.shape[0]

    # sort frequency vector in increasing order
    frequencies = wavelength_to_frequency(input_wavelength)
    indeces = np.argsort(frequencies)

    frequencies = frequencies[indeces]
    input_power = input_power[indeces]

    # Compute the frequency shifts for each signal
    frequency_shifts = np.zeros((num_signals, num_signals))
    for i in range(num_signals):
        frequency_shifts[i, :] = frequencies - frequencies[i]

    idx = 3

    # ! gain matrix is not symmetric here due to wrong frequency axis
    gains, spectrum, f = gain_spectrum(frequency_shifts, normalize=True)

    gains *= raman_coefficient / effective_core_area

    # Force diagonal to be 0
    np.fill_diagonal(gains, 0)
    gains = np.triu(gains) + np.triu(gains, 1).T

    # compute the frequency scaling factor
    freqs = np.expand_dims(frequencies, axis=-1)
    freq_scaling = np.maximum(1, freqs * (1/freqs.T))

    # gain matrix
    gain_matrix = freq_scaling * gains

    losses_linear = alpha_to_linear(losses)

    sol = scipy.integrate.odeint(
        raman_ode, input_power, z, args=(losses_linear, gain_matrix))

    if check_photon_count:
        photon_count = np.sum(sol / frequencies, axis=1)
        return sol[:, indeces], photon_count
    else:
        return sol[:, indeces]


def raman_ode(P, z, losses, gain_matrix):
    dPdz = np.zeros_like(P)
    dPdz = (-losses + np.matmul(gain_matrix,
                                P[:, np.newaxis])) * P[:, np.newaxis]
    return np.squeeze(dPdz)


if __name__ == "__main__":
    signal_wavelength = np.array([1550, 1560]) * 1e-9
    pump_wavelength = np.array([1470]) * 1e-9
    pump_power = np.array([600e-3])
    signal_power = 1e-6 * np.ones_like(signal_wavelength)

    powers = np.concatenate((signal_power, pump_power))
    wavelengths = np.concatenate((signal_wavelength, pump_wavelength))

    z = np.linspace(0, 10e3, 100)

    sol, photon_count = solve(powers, wavelengths, z,
                              losses=0.2, check_photon_count=True)

    output_spectrum = sol[-1, :]
    stem_bottom = 10*np.log10(np.min(output_spectrum)) - 10

    fig = plt.figure(1)
    plt.clf()
    fig, axs = plt.subplots(ncols=2, nrows=2, num=1)

    axs[0, 0].plot(z * 1e-3, 10*np.log10(sol))
    axs[0, 0].set_xlabel('Position [km]')

    axs[0, 1].stem(wavelengths * 1e9, 10*np.log10(output_spectrum),
                   bottom=stem_bottom)
    axs[0, 1].set_xlabel('Wavelength [\lambda]')

    axs[1, 0].plot(z * 1e-3, photon_count)
    axs[1, 0].set_xlabel('Position [km]')

    # plt.plot(z, 10*np.log10(sol[:, 2]))
    plt.grid(which='both')
    plt.show(block=False)
