import matplotlib.pyplot as plt
import numpy as np
import pytest
from pyraman import MMFAmplifier
from pyraman.sample_fibers import fiber


@pytest.mark.parametrize("counterpumping", [True, False])
def test_ase(counterpumping):
    """Test that signal/power solution are the same when computing ASE."""
    visual_debug = False  # Set to True to enable plotting
    amplifier_length = 100e3
    num_steps = 1000
    num_sigs = 100

    sig_power = 1e-5 * np.ones((num_sigs, fiber.modes))
    pump_power = 0.5 * 1e-3 if counterpumping else 100e-3
    pump_power = pump_power * np.ones((1, fiber.modes))

    sig_wave = np.linspace(1510e-9, 1590e-9, num_sigs)
    pump_wave = np.array([1430e-9])

    z = np.linspace(0, amplifier_length, num_steps)

    amplifier = MMFAmplifier()

    pump_solution_ase, signal_solution_ase, ase_solution = amplifier.solve(
        sig_power,
        sig_wave,
        pump_power,
        pump_wave,
        z,
        fiber=fiber,
        ase=True,
        counterpumping=counterpumping,
    )

    pump_solution, signal_solution = amplifier.solve(
        sig_power,
        sig_wave,
        pump_power,
        pump_wave,
        z,
        fiber=fiber,
        ase=False,
        counterpumping=counterpumping,
    )

    if not visual_debug:
        assert np.allclose(pump_solution, pump_solution_ase)
        assert np.allclose(signal_solution, signal_solution_ase)
        return

    pump_solution = 10 * np.log10(pump_solution) + 30
    signal_solution = 10 * np.log10(signal_solution) + 30
    ase_solution = 10 * np.log10(ase_solution) + 30

    linestyles = ["-", "--"]
    labels = ["LP01", "LP11"]

    plt.figure(1)
    plt.clf()
    for mode, (label, style) in enumerate(zip(labels, linestyles)):
        plt.plot(z * 1e-3, pump_solution[:, :, mode], linestyle=style, label=label)
        plt.plot(z * 1e-3, signal_solution[:, :, mode], linestyle=style, label=label)
    plt.xlabel("Position [km]")
    plt.ylabel("Power [dBm]")
    plt.grid()
    plt.xlim(0, z[-1] * 1e-3)
    plt.title("Signal and pump power evolution")

    plt.figure(2)
    plt.clf()
    for mode, (label, style) in enumerate(zip(labels, linestyles)):
        plt.plot(
            sig_wave * 1e9, signal_solution[-1, :, mode], linestyle=style, label=label
        )
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Power [dBm]")
    plt.title("Signal spectrum at fiber end")

    plt.figure(3)
    for mode, (label, style) in enumerate(zip(labels, linestyles)):
        plt.plot(z * 1e-3, ase_solution[:, :, mode], linestyle=style, label=label)
    plt.xlabel("Position [km]")
    plt.ylabel("Power [dBm]")
    plt.grid()
    plt.xlim(0, z[-1] * 1e-3)
    plt.title("ASE evolution")

    plt.figure(4)
    plt.clf()
    for mode, (label, style) in enumerate(zip(labels, linestyles)):
        plt.plot(
            sig_wave * 1e9, ase_solution[-1, :, mode], linestyle=style, label=label
        )
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Power [dBm]")
    plt.title("ASE spectrum at fiber end")

    plt.show()
