# -*- coding: utf-8 -*-
"""Created on Tue Jan 28 17:27:28 2020.

@author: Gianluca
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import polyval

from pyraman.solver.solvers import Fiber, RamanAmplifier
from pyraman.utils import alpha_to_linear, wavelength_to_frequency


class MMFAmplifier(RamanAmplifier):
    def __init__(self, bandwidth=40e12):

        super().__init__()

    def solve(
        self,
        signal_power,
        signal_wavelength,
        pump_power,
        pump_wavelength,
        z,
        fiber,
        counterpumping=False,
    ):
        """Solve the multi-mode Raman amplifier equations [1].

        Params
        ------
        signal_power: np.ndarray
            The input signal power in a 2d ndarray of shape (wavelengths, modes)
            expressed in Watt.
        signal_wavelength: np.ndarray
            The input signal wavelengths in a 2d ndarray of shape (wavelengths,)
            expressed in meters.
        pump_power: np.ndarray
            The input pump power in a 2d ndarray of shape (wavelengths, modes)
            expressed in Watt.
        pump_wavelength: np.ndarray
            The input pump wavelengths in a 2d ndarray of shape (wavelengths,)
            expressed in meters.
        z: np.ndarray
            The z-axis along which to integrate the equations, expressed in
            meters.
        fiber: pyraman.Fiber
            Fiber object defining the amplifier

        References
        ----------
        .. [1] Ryf, Roland, Rene Essiambre, Johannes von Hoyningen-Huene,
               and Peter Winzer. 2012. “Analysis of Mode-Dependent Gain in
               Raman Amplified Few-Mode Fiber.” In Optical Fiber Communication
               Conference, Los Angeles, California: OSA, OW1D.2.
               https://www.osapublishing.org/abstract.cfm?uri=OFC-2012-OW1D.2
               (June 5, 2019).
        """

        num_signals = signal_power.shape[0]
        num_pumps = pump_power.shape[0]

        total_wavelengths = num_signals + num_pumps

        total_signals = num_pumps + num_signals

        pump_power_ = pump_power.flatten()
        signal_power_ = signal_power.flatten()

        wavelengths = np.concatenate((pump_wavelength, signal_wavelength))
        input_power = np.concatenate((pump_power_, signal_power_))

        frequencies = wavelength_to_frequency(wavelengths)

        loss_coeffs = fiber.losses

        losses_ = polyval(loss_coeffs, wavelengths * 1e9)

        losses_linear = alpha_to_linear(losses_)
        losses_linear = np.repeat(losses_linear, fiber.modes)

        # Compute the frequency shifts for each signal
        frequency_shifts = np.zeros((total_signals, total_signals))
        for i in range(total_signals):
            frequency_shifts[i, :] = frequencies - frequencies[i]

        gains = self._interpolate_gain(frequency_shifts)

        gains *= fiber.raman_coefficient

        # Must be multiplied by overlap integrals

        # Force diagonal to be 0
        np.fill_diagonal(gains, 0)
        # gains = np.triu(gains) + np.triu(gains, 1).T

        # compute the frequency scaling factor
        freqs = np.expand_dims(frequencies, axis=-1)
        freq_scaling = np.maximum(1, freqs * (1 / freqs.T))

        oi = np.tile(fiber.overlap_integrals, (total_wavelengths, total_wavelengths))

        M = np.ones((fiber.modes, fiber.modes))

        # gain matrix
        gain_matrix = freq_scaling * gains

        gains_mmf = np.kron(gain_matrix, M) * oi

        direction = np.ones((total_wavelengths * fiber.modes,))
        if counterpumping:
            direction[: num_pumps * fiber.modes] = -1

        sol = scipy.integrate.odeint(
            MMFAmplifier.raman_ode,
            input_power,
            z,
            args=(losses_linear, gains_mmf, direction),
        )

        sol = sol.reshape((-1, total_signals, fiber.modes))

        pump_solution = sol[:, :num_pumps, :]
        signal_solution = sol[:, num_pumps:, :]

        return pump_solution, signal_solution

    @staticmethod
    def raman_ode(P, z, losses, gain_matrix, direction):
        """Integration step of the multimode Raman system."""
        dPdz = (-losses[:, np.newaxis] + np.matmul(gain_matrix, P[:, np.newaxis])) * P[
            :, np.newaxis
        ]

        return np.squeeze(dPdz) * direction


if __name__ == "__main__":
    amplifier = MMFAmplifier()

    overlap_integrals = np.array([[1, 1], [1, 2]])

    fiber = Fiber(modes=2, overlap_integrals=overlap_integrals)

    # sig_power = np.array([[1e-3, 2e-3]])

    num_sigs = 100

    sig_power = 1e-5 * np.ones((num_sigs, fiber.modes))
    pump_power = np.array([[100e-3, 100e-3]])

    sig_wave = np.linspace(1510e-9, 1590e-9, num_sigs)

    pump_wave = np.array([1430e-9])

    z = np.linspace(0, 100000, 1000)

    pump_solution, signal_solution = amplifier.solve(
        sig_power, sig_wave, pump_power, pump_wave, z, fiber=fiber
    )

    pump_solution = 10 * np.log10(pump_solution) + 30
    signal_solution = 10 * np.log10(signal_solution) + 30

    plt.figure(1)
    plt.clf()

    linestyles = ["-", "--"]
    labels = ["LP01", "LP11"]

    for mode, (label, style) in enumerate(zip(labels, linestyles)):
        plt.plot(z * 1e-3, pump_solution[:, :, mode], linestyle=style, label=label)
        plt.plot(z * 1e-3, signal_solution[:, :, mode], linestyle=style, label=label)

    plt.legend()
    plt.grid()

    plt.figure(2)
    plt.clf()

    for mode, (label, style) in enumerate(zip(labels, linestyles)):
        plt.plot(
            sig_wave * 1e9, signal_solution[-1, :, mode], linestyle=style, label=label
        )

    plt.show()
