import numpy as np
import scipy.integrate
from pyraman.response import gain_spectrum, impulse_response
from pyraman.utils import wavelength_to_frequency, alpha_to_linear
import matplotlib.pyplot as plt
from pdb import set_trace


class Fiber:
    def __init__(self, core_diameter=8e-6):

        self.effective_area = 80*(1e-6 ** 2)
        self.raman_coefficient = 7e-14

        print(self.raman_coefficient / self.effective_area)

        super().__init__()


class RamanAmplifier:

    def __init__(self, bandwidth=40e12, viz=False):

        self.bandwidth = bandwidth

        self.spline = self._compute_gain_spectrum(bandwidth)

        super().__init__()

    def _compute_gain_spectrum(self, bandwidth, spacing=50e9):

        fs = 2 * bandwidth

        num_samples = np.math.ceil(fs / spacing)

        dt = 1 / fs
        duration = (num_samples - 1) * dt
        dF = fs / num_samples

        f = np.arange(num_samples) * dF

        resp, t = impulse_response(None, fs, num_samples=num_samples)

        resp -= resp.mean()

        spectrum = np.fft.fft(resp)

        gain_spectrum = -np.imag(spectrum)

        # Normalize the peak to 1
        gain_spectrum /= np.max(np.abs(gain_spectrum))

        # Compute the spline representation of the signal for later use
        # This is done in the constructor so that each call of the
        # `propagate` method does not have to recompute it. It will only do
        # so in case the maximum frequency does not fall in the bandwidth
        # of the precomputed spectrum
        spline = scipy.interpolate.splrep(f, gain_spectrum, k=3)

        # if viz:
        #     plt.figure()
        #     plt.subplot(1, 2, 1)
        #     plt.plot(t * 1e12, resp)
        #     plt.xlabel('Time [ps]')

        #     plt.subplot(1, 2, 2)
        #     plt.plot(f * 1e-12, gain_spectrum, marker='.')
        #     # plt.yscale('log')
        #     plt.xlabel('Frequency shift [THz]')
        #     plt.show(block=False)

        return spline

    def _interpolate_gain(self, frequencies):

        # if negative frequencies are passed, just compute the gain
        # for the absolute value and then change the sign

        negative_idx = np.argwhere(frequencies < 0)

        pos_freqs = np.abs(frequencies)

        max_freq = np.max(pos_freqs)

        if max_freq > self.bandwidth:
            print("Warning: recomputing spline representation")
            self.bandwidth = 2 * max_freq

            # Recompute the gain spectrum with a bigger bandwidth
            self.spline = self._compute_gain_spectrum(self.bandwidth, viz=True)

        # compute the interpolate values from the spline representation
        # we computed in the constructor
        gains = scipy.interpolate.splev(pos_freqs, self.spline)

        # switch the sign for negative frequencies
        gains[negative_idx] *= -1

        return gains

    def propagate(self):
        pass

    def solve(self, input_power, input_wavelength, z, losses=0.2, raman_coefficient=1,
              effective_core_area=80e-12, check_photon_count=False):

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

        gains = self._interpolate_gain(frequency_shifts)

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

    @staticmethod
    def raman_ode(P, z, losses, gain_matrix):
        dPdz = np.zeros_like(P)
        dPdz = (-losses + np.matmul(gain_matrix,
                                    P[:, np.newaxis])) * P[:, np.newaxis]
        return np.squeeze(dPdz)


if __name__ == "__main__":
    ampl = RamanAmplifier(bandwidth=40e12, viz=True)

    freqs = np.linspace(-30, 30, 1000) * 1e12
    gains = ampl._interpolate_gain(freqs)

    plt.figure()
    plt.plot(freqs, gains)
    plt.show(block=False)
    # signal_wavelength = np.array([1550, 1560]) * 1e-9
    # pump_wavelength = np.array([1470]) * 1e-9
    # pump_power = np.array([600e-3])
    # signal_power = 1e-6 * np.ones_like(signal_wavelength)

    # powers = np.concatenate((signal_power, pump_power))
    # wavelengths = np.concatenate((signal_wavelength, pump_wavelength))

    # z = np.linspace(0, 10e3, 100)

    # sol, photon_count = solve(powers, wavelengths, z,
    #                           losses=0.2, check_photon_count=True)

    # output_spectrum = sol[-1, :]
    # stem_bottom = 10*np.log10(np.min(output_spectrum)) - 10

    # fig = plt.figure(1)
    # plt.clf()
    # fig, axs = plt.subplots(ncols=2, nrows=2, num=1)

    # axs[0, 0].plot(z * 1e-3, 10*np.log10(sol))
    # axs[0, 0].set_xlabel('Position [km]')

    # axs[0, 1].stem(wavelengths * 1e9, 10*np.log10(output_spectrum),
    #                bottom=stem_bottom)
    # axs[0, 1].set_xlabel('Wavelength [\lambda]')

    # axs[1, 0].plot(z * 1e-3, photon_count)
    # axs[1, 0].set_xlabel('Position [km]')

    # # plt.plot(z, 10*np.log10(sol[:, 2]))
    # plt.grid(which='both')
    # plt.show(block=False)
