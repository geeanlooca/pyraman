import matplotlib.pyplot as plt
import numpy as np
import pytest
from pyraman import MMFAmplifier
from pyraman.sample_fibers import fiber
from pyraman.solver._mmf import TimeOut


def test_shooting():
    """Test that shooting method works."""
    visual_debug = False  # Set to True to enable plotting
    amplifier_length = 100e3
    num_steps = 1000
    num_sigs = 100
    num_pumps = 8

    sig_power = 1e-5 * np.ones((num_sigs, fiber.modes))
    pump_power = 400e-3
    pump_power = pump_power * np.ones((num_pumps, fiber.modes))

    sig_wave = np.linspace(1510e-9, 1590e-9, num_sigs)
    pump_wave = np.linspace(1450e-9, 1480e-9, num_pumps)

    z = np.linspace(0, amplifier_length, num_steps)

    amplifier = MMFAmplifier()

    with pytest.raises(TimeOut):
        pump_solution, signal_solution = amplifier.solve(
            sig_power,
            sig_wave,
            pump_power,
            pump_wave,
            z,
            fiber=fiber,
            ase=False,
            shooting=True,
        )
