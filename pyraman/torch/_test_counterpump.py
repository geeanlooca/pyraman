"""Module for testing the Raman Multimode PyTorch solver in counterpumping mode."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from pyraman import MMFAmplifier
from pyraman.sample_fibers import fiber
from pyraman.torch import MMFRamanAmplifier


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--benchmark", default=False, action="store_true")
    parser.add_argument(
        "-s",
        "--steps",
        default=200,
        type=int,
        help="The number of integration steps for the PyTorch RK4 integrator.",
    )

    parser.add_argument(
        "-c",
        "--channels",
        default=100,
        type=int,
        help="The number of signal wavelengths to amplify.",
    )

    parser.add_argument(
        "-P",
        "--power-per-channel",
        type=float,
        default=1e-4,
        help="The power in each signal mode.",
    )

    parser.add_argument(
        "-p", "--pumps", default=2, type=int, help="The number of pump wavelengths."
    )

    parser.add_argument(
        "-t",
        "--tries",
        default=5,
        type=int,
        help="The number of runs for each batch size",
    )

    parser.add_argument(
        "-L",
        "--length",
        type=float,
        default=70e3,
        help="The length of the fiber in meters.",
    )

    args = parser.parse_args()

    num_pumps = args.pumps
    num_channels = args.channels
    power_per_channel = args.power_per_channel

    length = args.length
    steps = 1000
    steps_t = args.steps
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
        counterpumping=True,
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

    plt.figure(1)
    plt.plot(signal_lambda * 1e9, onoff_gain, marker=".")
    plt.plot(signal_lambda * 1e9, onoff_gain_torch, "-o", fillstyle="none")
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Gain [dB]")

    if not args.benchmark:
        plt.show()
    else:
        if device != "cuda":
            raise SystemError("Benchmarking is only supported using CUDA.")

        batch_sizes = [100, 200, 300, 500, 700, 800, 1000, 2000, 2500]

        tries = args.tries
        duration = np.zeros((len(batch_sizes,)))
        iters_per_second = np.zeros_like(duration)
        error = np.zeros_like(duration)

        P = torch.zeros((2, 3))

        print("\nBenchmarking...")
        for x, batch_size in enumerate(batch_sizes):

            # Prepare the input Tensors
            pump_lambda_t = (
                torch.rand((num_pumps,), device=device, dtype=torch.float32) * 60 + 1420
            ) * 1e-9
            pump_power_t = (
                torch.rand((num_pumps, fiber.modes), device=device, dtype=torch.float32)
            ) * 300e-3

            torch_input = torch.cat((pump_lambda_t, pump_power_t.view((-1,))), 0).view(
                1, -1
            )
            torch_input = torch_input.repeat(batch_size, 1)

            # Initialize cuda timers
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            durations = []

            for try_ in range(tries):
                start.record()
                out = raman_model(torch_input)
                end.record()
                torch.cuda.synchronize()
                durations.append(start.elapsed_time(end) * 1e-3)

            duration[x] = np.mean(durations)
            error[x] = np.std([batch_size / d for d in durations])
            iters_per_second[x] = batch_size / duration[x]

            print(f"{batch_size}: \t{iters_per_second[x]} It/s")

        plt.figure()
        plt.subplot(121)
        plt.plot(batch_sizes, duration * 1e3, "-x")
        plt.xlabel("Batch size")
        plt.title("Duration [ms]")
        plt.subplot(122)
        plt.errorbar(batch_sizes, iters_per_second, yerr=2 * error)
        plt.xlabel("Batch size")
        plt.title("Iterations / second")
        plt.show()
