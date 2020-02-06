# -*- coding: utf-8 -*-
"""Created on Tue Jan 28 17:27:28 2020.

@author: Gianluca
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy

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
        direction=1,
        maxiter=100,
        tol=1e-3
    ):
        """Solve the multi-mode Raman amplifier equations [1][2].

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
        direction: np.ndarray or scalar
            Array definining the propagation direction of the pumps. 
            The sign of every element determines if the respective pump
            is co-propagating (positive) or counter-propagating (negative). 
            If scalar, it affects every pump.
        maxiter: int
            In case of counter-pumping amplifier, the maximum number of
            iterations allowed for the shooting algorithm.
        tol: float
            Error tolerance on the pumps initial value when considering
            a counter-pumping amplifier. If the error on every pump is below
            `tol`, the shooting algorithm stops. Naturally defined in the same
            units as `pump_power`.
            

        Notes
        -----
        All the quantities are stored alternating modes first, and then
        wavelength.

        Raises
        ------
        ValueError
            when the `direction` parameter is not a scalar or has a number
            of elements different than the number of pump wavelengths.

        References
        ----------
        .. [1] Ryf, Roland, Rene Essiambre, Johannes von Hoyningen-Huene,
               and Peter Winzer. 2012. “Analysis of Mode-Dependent Gain in
               Raman Amplified Few-Mode Fiber.” In Optical Fiber Communication
               Conference, Los Angeles, California: OSA, OW1D.2.
               https://www.osapublishing.org/abstract.cfm?uri=OFC-2012-OW1D.2
               (June 5, 2019).
        .. [2] Zhou, Junhe. 2014. “An Analytical Approach for Gain Optimization
               in Multimode Fiber Raman Amplifiers.” Optics Express
               22(18): 21393.

        """

        num_signals = signal_power.shape[0]
        num_pumps = pump_power.shape[0]

        total_wavelengths = num_signals + num_pumps

        total_signals = num_pumps + num_signals

        pump_power_ = pump_power.flatten()
        signal_power_ = signal_power.flatten()

        # Handle the counter-pumping case
        direction_ = np.ones((total_signals,))
        
        if np.size(direction) == 1:
            direction_[:num_pumps] *= np.sign(direction)
        elif np.size(direction) == num_pumps:
            direction_[:num_pumps] = np.sign(direction)
        else:
            raise ValueError(
                "direction must either be a scalar or a vector"
                + "of the same size as pump_wavelengths"
            )

        direction_ = np.repeat(direction_, fiber.modes)

        wavelengths = np.concatenate((pump_wavelength, signal_wavelength))
        input_power = np.concatenate((pump_power_, signal_power_))

        frequencies = wavelength_to_frequency(wavelengths)

        losses_linear = alpha_to_linear(fiber.losses_dB)

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

        if np.any(direction_ < 0):
            # Shooting is needed

            # Target pump values
            bw_idx = direction_ < 0
            bw_pump_power = input_power[bw_idx]

            # The initial pump guess is the scaled undepleted pump solution
            guess = bw_pump_power * np.exp(-z[-1] * losses_linear) / 10

            # Error tolerance: stop when every guess differs from the target
            # values less than `tol`
            tol = 1e-3

            # Number of iterations
            iters = 0

            # Start the loop
            stop = False
            
            scaling = 0.1
            
            while not stop and iters < maxiter:

                # Replace the initial pump values with the current guess
                input_power[bw_idx] = guess

                sol = scipy.integrate.odeint(
                    MMFAmplifier.raman_ode,
                    input_power,
                    z,
                    args=(losses_linear, gains_mmf, direction_),
                )

                # Compute the residual error
                curr_value = sol[-1, bw_idx]
                residual = curr_value - bw_pump_power

                # If the error is below the tolerance, stop
                if np.all(np.abs(residual) < tol):
                    stop = True

                iters += 1

                # Update rule
                guess = np.maximum(0, guess - scaling * residual)

            print(f"Iterations: {iters}")
            
            if iters >= maxiter:
                print(f"Reached maximum number of iterations.")

        else:
            # Forward pumping
            sol = scipy.integrate.odeint(
                MMFAmplifier.raman_ode,
                input_power,
                z,
                args=(losses_linear, gains_mmf, direction_),
            )

        # Reshape the solution so that the third dimension specifies the
        # mode.
        sol = sol.reshape((-1, total_signals, fiber.modes))

        pump_solution = sol[:, :num_pumps, :]
        signal_solution = sol[:, num_pumps:, :]

        return pump_solution, signal_solution

    @staticmethod
    def raman_ode(P, z, losses, gain_matrix, direction):
        """Integration step of the multimode Raman system."""
        dPdz = (-losses + np.matmul(gain_matrix, P[:, np.newaxis])) * P[:, np.newaxis]

        return direction * np.squeeze(dPdz)


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
