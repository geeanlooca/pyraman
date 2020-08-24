# -*- coding: utf-8 -*-
"""Created on Tue Jan 28 17:27:28 2020.

@author: Gianluca
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import polyval
import scipy.optimize
from scipy.constants import Boltzmann as kB
from scipy.constants import Planck as h_planck
from scipy.constants import lambda2nu

from pyraman.solver.solvers import RamanAmplifier
from pyraman.utils import alpha_to_linear, wavelength_to_frequency


class TimeOut(Exception):
    """Raise a timeout exception to stop the shooting algorithm."""

    pass


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
        ase=False,
        reference_bandwidth=0.1,
        temperature=300,
        shooting=None,
        initial_guesses=None,
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
        counterpumping: bool, optional
            If True, `pump_power` is considered to be the pump power
            at the start of the fiber, i.e. at z = 0. The signs
            of the equations for the pump power evolution are set to -1,
            so the losses actually amplify the pump power during
            propagation in the z: 0->L direction. Optional, by default False.
        reference_bandwidth: float, optional
            The reference optical bandwidth (nm) for ASE measurement.
            Optional, by default 0.1 nm.
        temperature: float, optional
            The optical fiber temperature (K). Optional, by default 300 K.
        shooting : bool, optional
            Use a shooting method to solve the counterpumping case,
            by default None.

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

        if not ase:
            direction = np.ones((total_wavelengths * fiber.modes,))

            if counterpumping or shooting:
                direction[: num_pumps * fiber.modes] = -1

            if not shooting:
                sol = scipy.integrate.odeint(
                    MMFAmplifier.raman_ode,
                    input_power,
                    z,
                    args=(losses_linear, gains_mmf, direction),
                )

                sol = sol.reshape((-1, total_signals, fiber.modes))

                pump_solution = sol[:, :num_pumps, :]
                signal_solution = sol[:, num_pumps:, :]
            else:
                # SHOOTING METHOD

                pump_losses = losses_linear[: num_pumps * fiber.modes]

                if initial_guesses is None:
                    initial_guesses = pump_power_ * np.exp(-z[-1] * pump_losses) / 10

                callback_info = {
                    "max_error": float("inf"),
                    "iter": 0,
                    "threshold": 0.5,
                    "params": None,
                }

                def callback(x):
                    max_error = callback_info["max_error"]
                    threshold = callback_info["threshold"]
                    print(f"Max error: {max_error} mW")
                    if max_error < threshold:
                        # print(f"Shooting error < {threshold}, stopping optimization")
                        callback_info["params"] = np.copy(x)
                        raise TimeOut("Stopping optimization")

                def optim_fun(x0):
                    input_power = np.concatenate((x0, signal_power_))

                    sol = scipy.integrate.odeint(
                        MMFAmplifier.raman_ode,
                        input_power,
                        z,
                        args=(losses_linear, gains_mmf, direction),
                    )

                    sol = sol.reshape((-1, total_signals, fiber.modes))

                    pump_solution = sol[-1, :num_pumps, :].flatten()

                    # return the MSE between desired solution and obtained values
                    cost = np.sqrt(
                        np.mean((pump_solution * 1e3 - pump_power_ * 1e3) ** 2)
                    )

                    max_error = np.max(np.abs(pump_solution - pump_power_)) * 1e3

                    callback_info["max_error"] = max_error

                    # print(f"RMSE: {cost:.3f} mW\tMax. Error {max_error:.2f} mW")

                    return cost

                bounds = [(0, None) for _ in range(num_pumps * fiber.modes)]

                try:
                    result = scipy.optimize.minimize(
                        # optim_fun, initial_guesses, method="L-BFGS-B", bounds=bounds
                        optim_fun,
                        initial_guesses,
                        method="L-BFGS-B",
                        options={"maxfun": 1e8, "maxiter": 1e6, "ftol": 1e-11},
                        bounds=bounds,
                        callback=callback,
                    )

                    x0 = result.x
                except TimeOut:
                    x0 = callback_info["params"]

                # Result of the optimization process: pump power at z=0

                # Propagate
                input_power = np.concatenate((x0, signal_power_))

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
        else:
            direction = np.ones(((total_wavelengths + num_signals) * fiber.modes,))

            # Compute the phonon occupancy factor
            Hinv = np.exp(h_planck * np.abs(frequency_shifts) / (kB * temperature)) - 1
            eta = 1 + 1 / Hinv
            np.fill_diagonal(eta, 0)
            eta = np.repeat(np.repeat(eta, fiber.modes, axis=0), fiber.modes, axis=1)

            # Compute the new Raman gain matrix
            gain_matrix_ase = eta * gains_mmf

            # Convert reference bandwidth in hertz using the
            # central signal wavelength as reference
            central_wavelength = (signal_wavelength.max() + signal_wavelength.min()) / 2
            reference_bandwidth *= 1e-9  # Convert in meters
            w_a = central_wavelength - reference_bandwidth / 2
            w_b = central_wavelength + reference_bandwidth / 2
            f_a = lambda2nu(w_a)
            f_b = lambda2nu(w_b)
            reference_bandwidth_hz = np.abs(f_a - f_b)
            print(reference_bandwidth * 1e9, " nm")
            print(reference_bandwidth_hz * 1e-9, " GHz")

            if counterpumping or shooting:
                direction[: num_pumps * fiber.modes] = -1

            # Initial conditions, ase power must be 0 at z=0
            input_power_ase = np.zeros((input_power.size + num_signals * fiber.modes,))
            input_power_ase[: input_power.size] = input_power

            signal_frequencies = wavelength_to_frequency(signal_wavelength)
            ase_frequencies = np.repeat(signal_frequencies, fiber.modes)

            sol = scipy.integrate.odeint(
                MMFAmplifier.raman_ode_with_ase,
                input_power_ase,
                z,
                args=(
                    losses_linear,
                    gains_mmf,
                    gain_matrix_ase,
                    ase_frequencies,
                    reference_bandwidth_hz,
                    direction,
                    num_signals,
                    num_pumps,
                    fiber.modes,
                ),
            )

            sol = sol.reshape((-1, total_signals + num_signals, fiber.modes))

            power_solution = sol[:, :total_signals, :]
            pump_solution = power_solution[:, :num_pumps, :]
            signal_solution = power_solution[:, num_pumps:, :]
            ase_solution = sol[:, -num_signals:, :]

            return pump_solution, signal_solution, ase_solution

    def solve_shooting():
        pass

    @staticmethod
    def raman_ode(P, z, losses, gain_matrix, direction):
        """Integration step of the multimode Raman system."""
        dPdz = (-losses[:, np.newaxis] + np.matmul(gain_matrix, P[:, np.newaxis])) * P[
            :, np.newaxis
        ]

        return np.squeeze(dPdz) * direction

    def raman_ode_with_ase(
        P,
        z,
        losses,
        gain_matrix,
        gain_matrix_ase,
        frequencies,
        ref_bandwidth,
        direction,
        num_signals,
        num_pumps,
        num_modes,
    ):
        """Integration step of the multimode Raman system with ASE."""
        num_ase = num_signals * num_modes
        num_power = (num_signals + num_pumps) * num_modes

        P_ = P[:num_power]
        P_ase = P[-num_ase:]
        losses_ase = losses[-num_ase:]

        gain_factor = np.matmul(gain_matrix, P_[:, np.newaxis])

        dPowerdz = (-losses[:, np.newaxis] + gain_factor) * P_[:, np.newaxis]

        gain_factor_ase = np.matmul(gain_matrix_ase, P_[:, np.newaxis])
        gain_factor_ase = gain_factor_ase[-num_ase:]

        dASEdz = -losses_ase[:, np.newaxis] * P_ase[:, np.newaxis]
        dASEdz += gain_factor[-num_ase:] * P_ase[:, np.newaxis]
        dASEdz += (
            gain_factor_ase * 2 * h_planck * frequencies[:, np.newaxis] * ref_bandwidth
        )

        dPdz = np.vstack((dPowerdz, dASEdz))
        return np.squeeze(dPdz) * direction
