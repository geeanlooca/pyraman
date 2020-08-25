"""Module for testing the Raman Multimode PyTorch solver."""

import argparse

import pytest

import matplotlib.pyplot as plt
import numpy as np

import torch
from pyraman import MMFAmplifier
from pyraman.sample_fibers import fiber
from pyraman.torch import MMFRamanAmplifier
from pyraman.torch import MMFTorchRamanAmplifierWithASE


def test_torch_solution_vs_numpy():
    visual_debug = False  # Set to True to enable plotting
    device = "cpu"

    num_pumps = 4
    num_channels = 50
    power_per_channel = 0.1e-3

    length = 70e3
    steps = 1000
    steps_t = 200
    z = np.linspace(0, length, steps)

    signal_lambda = np.linspace(1520, 1570, num_channels) * 1e-9
    signal_power = np.ones((num_channels, fiber.modes)) * power_per_channel

    pump_power = np.random.uniform(0, 300e-3, (num_pumps, fiber.modes))
    pump_lambda = np.random.uniform(1430e-9, 1470e-9, num_pumps)

    raman_model = MMFRamanAmplifier(
        length,
        steps_t,
        num_pumps,
        torch.from_numpy(signal_lambda).float(),
        power_per_channel,
        fiber,
        dF=50e9,
    )
    raman_model.to(device)

    # The standard solver
    amplifier = MMFAmplifier()

    # Compute the on-off gain
    _, signal_off = amplifier.solve(
        signal_power, signal_lambda, 0 * pump_power, pump_lambda, z, fiber
    )
    _, signal_solution = amplifier.solve(
        signal_power, signal_lambda, pump_power, pump_lambda, z, fiber
    )
    onoff_gain = 10 * np.log10(signal_solution[-1] / signal_off[-1])

    # Prepare input to the PyTorch solver
    pump_lambda_t = torch.from_numpy(pump_lambda).float().to(device)
    pump_power_t = torch.from_numpy(pump_power).float().to(device)
    torch_input = torch.cat((pump_lambda_t, pump_power_t.view((-1,))), 0).view(1, -1)

    # ... and compute the on-off gain
    torch_solution_on = raman_model(torch_input).view(num_channels, fiber.modes)
    torch_solution_off = raman_model.off_gain.view(num_channels, fiber.modes)

    onoff_gain_torch = (
        10 * torch.log10(torch_solution_on / torch_solution_off).cpu().numpy()
    )

    # Compute the error between the two solutions
    rmse = np.mean((onoff_gain_torch - onoff_gain) ** 2, axis=0) ** 0.5
    print(f"RMSE: {rmse} dB")

    if not visual_debug:
        # Check that the two solutions are off by at most 0.01 dB
        assert np.allclose(onoff_gain, onoff_gain_torch, atol=0.02)
    else:

        plt.figure(1)
        plt.plot(signal_lambda * 1e9, onoff_gain, marker=".", label="Numpy solution")
        plt.plot(
            signal_lambda * 1e9,
            onoff_gain_torch,
            "-o",
            fillstyle="none",
            label="Torch solution",
        )
        plt.xlabel("Wavelength [nm]")
        plt.ylabel("Gain [dB]")
        plt.legend()

        plt.show()


@pytest.mark.parametrize("counterpumping", [True, False])
@pytest.mark.parametrize("num_pumps", [1, 2, 4])
def test_torch_ase_solution_vs_numpy(counterpumping, num_pumps):
    device = "cpu"
    visual_debug = False  # Set to True to enable plotting
    amplifier_length = 100e3
    num_steps = 1000
    num_channels = 100
    power_per_channel = 0.1e-3
    steps = 500

    sig_power = power_per_channel * np.ones((num_channels, fiber.modes))
    pump_power = 0.5 * 1e-3 if counterpumping else 100e-3
    pump_power = pump_power * np.ones((num_pumps, fiber.modes))

    sig_wave = np.linspace(1510e-9, 1590e-9, num_channels)
    pump_wave = (
        np.array([1430e-9])
        if num_pumps == 1
        else np.linspace(1430, 1490, num_pumps) * 1e-9
    )

    z = np.linspace(0, amplifier_length, num_steps)

    amplifier = MMFAmplifier()

    pump_solution_ase_off, signal_solution_ase_off, ase_solution_off = amplifier.solve(
        sig_power,
        sig_wave,
        pump_power * 0,
        pump_wave,
        z,
        fiber=fiber,
        ase=True,
        counterpumping=counterpumping,
        reference_bandwidth=0.5,
    )

    pump_solution_ase, signal_solution_ase, ase_solution = amplifier.solve(
        sig_power,
        sig_wave,
        pump_power,
        pump_wave,
        z,
        fiber=fiber,
        ase=True,
        counterpumping=counterpumping,
        reference_bandwidth=0.5,
    )

    gain_numpy = 10 * np.log10(signal_solution_ase[-1] / signal_solution_ase_off[-1])

    signal_power = np.ones((num_channels, fiber.modes)) * power_per_channel

    raman_model = MMFTorchRamanAmplifierWithASE(
        amplifier_length,
        steps,
        num_pumps,
        torch.from_numpy(sig_wave).float(),
        power_per_channel,
        fiber,
        dF=50e9,
        counterpumping=counterpumping,
        reference_bandwidth=0.5,
    )
    raman_model.to(device)

    # Prepare input to the PyTorch solver
    pump_lambda_t = torch.from_numpy(pump_wave).float().to(device)
    pump_power_t = torch.from_numpy(pump_power).float().to(device)
    torch_input = torch.cat((pump_lambda_t, pump_power_t.view((-1,))), 0).view(1, -1)

    # ... and compute the on-off gain
    if counterpumping:
        torch_spectrum_on, torch_ase_on, _ = raman_model(torch_input)
    else:
        torch_spectrum_on, torch_ase_on = raman_model(torch_input)

    torch_solution_on = torch_spectrum_on.view(num_channels, fiber.modes)
    torch_solution_off = raman_model.off_gain.view(num_channels, fiber.modes)

    ase_torch = 10 * torch.log10(torch_ase_on).cpu().numpy().squeeze()
    ase_numpy = 10 * np.log10(ase_solution[-1])

    onoff_gain_torch = (
        10 * torch.log10(torch_solution_on / torch_solution_off).cpu().numpy()
    )

    if not visual_debug:
        assert np.allclose(gain_numpy, onoff_gain_torch, atol=0.02)
        assert np.allclose(ase_numpy, ase_torch, atol=0.02)
    else:
        plt.figure(1)
        plt.plot(sig_wave * 1e9, gain_numpy, marker=".", label="Numpy solution")

        plt.plot(
            sig_wave * 1e9,
            onoff_gain_torch,
            "-o",
            fillstyle="none",
            label="Torch solution",
        )

        plt.xlabel("Wavelength [nm]")
        plt.ylabel("Gain [dB]")
        plt.legend()

        plt.figure(2)
        plt.plot(sig_wave * 1e9, ase_numpy, marker=".", label="Numpy solution")

        plt.plot(
            sig_wave * 1e9, ase_torch, "-o", fillstyle="none", label="Torch solution",
        )

        plt.xlabel("Wavelength [nm]")
        plt.ylabel("ASE Spectrum [dB]")
        plt.legend()

        plt.show()
