import math

import numpy as np
import torch
from scipy.constants import speed_of_light

from pyraman.response import impulse_response
from pyraman.torch import torch_rk4


class MMFRamanAmplifier(torch.nn.Module):
    def __init__(
        self,
        length,
        steps,
        num_pumps,
        signal_wavelength,
        power_per_channel,
        fiber,
        fs=80e12,
        dF=50e9,
    ):
        """A PyTorch model of a Multi-Mode Fiber Raman Amplifier.

        Parameters
        ----------
        length : float
            The length of the fiber [m].
        steps : int
            The number of integration steps.
        num_pumps : int
            The number of Raman pumps.
        signal_wavelength : torch.Tensor
            The input signal wavelenghts.
        power_per_channel : float
            The input power for each channel (wavelength/mode).
        fiber : pyraman.Fiber
            The Fiber object containing relevant parameters of the fiber.
        fs : float, optional
            The sampling frequency of the Raman impulse response, by default 80e12
        dF : float, optional
            The spacing between samples of the Raman frequency response, by default 50e9
        """

        super(MMFRamanAmplifier, self).__init__()
        self.c0 = speed_of_light
        self.power_per_channel = power_per_channel
        self.num_pumps = num_pumps
        self.num_channels = signal_wavelength.shape[0]
        self.modes = fiber.modes
        self.length = length
        self.steps = steps
        z = torch.linspace(0, self.length, self.steps)

        signal_power = self.power_per_channel * torch.ones(
            (1, self.num_channels * self.modes)
        )

        # limit the polynomial fit of the attenuation spectrum to order 2
        num_loss_coeffs = len(fiber.losses)
        loss_coeffs = np.zeros((3,))

        for i in range(np.minimum(3, num_loss_coeffs)):
            loss_coeffs[i] = fiber.losses[i]

        # Compute the attenuation for each signal
        signal_loss = self._alpha_to_linear(
            loss_coeffs[2]
            + loss_coeffs[1] * (signal_wavelength * 1e9)
            + loss_coeffs[0] * (signal_wavelength * 1e9) ** 2
        )

        signal_loss = signal_loss.repeat_interleave(self.modes).view(1, -1)

        overlap_integrals = torch.Tensor(fiber.overlap_integrals)

        self.raman_coefficient = fiber.raman_coefficient

        # Compute the raman gain spectrum with the given precision
        self.fs = fs
        self.dF = dF
        N = math.ceil(self.fs / self.dF)
        self.dF = self.fs / N
        resp, _ = impulse_response(None, self.fs, num_samples=N)
        resp -= resp.mean()

        # Only save the positive spectrum
        self.response_length = math.ceil((N + 1) / 2)
        self.max_freq = (self.response_length - 1) * self.dF
        spectrum = np.fft.fft(resp)[: self.response_length]
        gain_spectrum = -np.imag(spectrum)

        # Normalize the peak to the raman coefficient of the fiber
        gain_spectrum /= np.max(np.abs(gain_spectrum))
        gain_spectrum *= self.raman_coefficient

        # Transform it to torch.Tensor, reshape it, and register as a buffer
        raman_response = torch.from_numpy(gain_spectrum).float()
        # Reshape the Tensor in a format accepted by grid_sample
        # It works on 4D (batch of multi-channel images) or 5D (volumetric data)
        # So Tensor must be of size (self.batch_size, 1, 1, response_length), meaning
        # we encode the response as a 1-channel, 1 pixel tall image
        raman_response = raman_response.view(1, 1, 1, -1)

        # Register buffers to make it work on a GPU
        self.register_buffer("signal_power", signal_power)
        self.register_buffer("z", z)
        self.register_buffer(
            "signal_frequency", self._lambda2frequency(signal_wavelength)
        )
        self.register_buffer("loss_coefficients", torch.from_numpy(loss_coeffs).float())
        self.register_buffer("signal_loss", signal_loss)
        self.register_buffer("overlap_integrals", overlap_integrals)
        self.register_buffer("raman_response", raman_response)

        # Doesn't matter, the pumps are turned off
        pump_lambda = torch.linspace(1420, 1480, self.num_pumps) * 1e-9
        pump_power = torch.zeros((num_pumps * self.modes))
        x = torch.cat((pump_lambda, pump_power)).float().view(1, -1)

        # Propagate the pumps to compute the output spectrum
        off_gain = self.forward(x).view(1, -1)

        # Save it in a buffer
        self.register_buffer("off_gain", off_gain)

    def _alpha_to_linear(self, alpha):
        """Convert attenuation constant from dB to linear units."""
        return alpha * 1e-3 * np.log(10) / 10

    def _lambda2frequency(self, wavelength):
        """Convert wavelength in frequency."""
        return self.c0 / wavelength

    def _batch_diff(self, x):
        """Takes a Tensor of shape (B, N) and returns a Tensor of shape (B, N,
        N) where in position (i, :, :) is the matrix of differences of the
        input vector (i, :) with each one of its elements."""
        batch_size = x.shape[0]
        D = x.view(batch_size, 1, -1) - x.view(batch_size, -1, 1)
        return D

    def _interpolate_response(self, freqs):
        """Compute the Raman gain coefficient for the input frequencies."""
        batch_size = freqs.shape[0]

        # The input for `grid_sample` are defined on a [-1, 1] axis
        # so we need to normalize the input frequencies accordingly
        norm_freqs = 2 * freqs / self.max_freq - 1

        return torch.nn.functional.grid_sample(
            self.raman_response.expand(batch_size, 1, 1, self.response_length),
            norm_freqs,
            align_corners=True,
        )

    @staticmethod
    def ode(P, z, losses, gain_matrix):
        """Batched version of the multimode Raman amplifier equations.

        Params
        ------
        P : torch.Tensor
            Size (B, M * S). Power of the signal on every mode-wavelength combination
        z : torch.Tensor
            Size (Z,) The evaluation point.
        losses : torch.Tensor.
            Size (B, M * S)
        gain_matrix : torch.Tensor
            Size (B, M*S, M*S)

        Returns
        -------
        torch.Tensor
            Propagated power values
        """

        batch_size = P.shape[0]
        dPdz = (
            -losses.view(batch_size, -1, 1)
            + torch.matmul(gain_matrix, P.view(batch_size, -1, 1))
        ) * P.view(batch_size, -1, 1)

        return dPdz.view(batch_size, -1)

    def forward(self, x):
        """Solves the propagation equation using a RK4 scheme.

        Parameters
        ----------
        x : torch.Tensor
            pump wavelength and pump power, size (N_batch, N_pumps * (N_modes + 1))

        Returns
        -------
        torch.Tensor
            Gain on each mode (B, N_signals, N_modes)
        """

        # import pdb
        # pdb.set_trace()

        batch_size = x.shape[0]

        num_freqs = self.num_channels + self.num_pumps

        # This will be the input to the interpolation function
        interpolation_grid = torch.zeros(
            (batch_size, 1, num_freqs ** 2, 2), dtype=x.dtype, device=x.device,
        )

        pump_wavelengths = x[:, : self.num_pumps]

        # Compute the loss for each pump wavelength/mode
        pump_loss = self._alpha_to_linear(
            self.loss_coefficients[2]
            + self.loss_coefficients[1] * pump_wavelengths * 1e9
            + self.loss_coefficients[0] * (pump_wavelengths * 1e9) ** 2
        ).repeat_interleave(self.modes, dim=1)

        # Concatenate the pump losses to the signal losses
        losses = torch.cat(
            (
                pump_loss,
                self.signal_loss.expand(batch_size, self.num_channels * self.modes),
            ),
            dim=1,
        )

        # Concatenate input pump wavelengths with signal wavelengths in a (B, P + S)
        # Tensor
        pump_freqs = self._lambda2frequency(pump_wavelengths)

        total_freqs = torch.cat(
            (pump_freqs, self.signal_frequency.expand(batch_size, self.num_channels)),
            dim=1,
        )

        # Concatenate input pump power and signal power
        total_power = torch.cat(
            (
                x[:, self.num_pumps :],
                self.signal_power.expand(batch_size, self.num_channels * self.modes),
            ),
            1,
        )

        # Compute the difference in frequencies
        freqs_diff = self._batch_diff(total_freqs)

        # Collapse matrix of differences in a vector
        interpolation_grid[:, 0, :, 0] = freqs_diff.view(batch_size, -1)

        # Compute the raman gain between the signals and pumps
        # ! Should I precompute the raman gain between signals?
        # ! I could compute it in the __init__ and then expand/cat
        # ! when needed.
        idx = interpolation_grid[:, 0, :, 0] < 0
        gain = self._interpolate_response(torch.abs(interpolation_grid))[:, 0, 0, :]
        gain[idx] *= -1

        # import pdb

        # pdb.set_trace()

        # Restore the batched matrices: (B, N * N) -> (B, N, N)
        gain = gain.view(batch_size, num_freqs, num_freqs)
        # diag = torch.diagonal(gain, dim1=-2, dim2=-1).fill_(0)

        # Compute the scaling factor
        one = torch.tensor(1, dtype=x.dtype, device=x.device)
        freq_scaling = torch.max(
            one,
            total_freqs.view(batch_size, -1, 1)
            / (total_freqs.view(batch_size, -1, 1).transpose(1, 2)),
        )

        gain *= freq_scaling

        # build gain matrix, tiling gains by the number of modes and multiplying
        # for the corresponding overlap integral

        # In the gain matrix, each entry corresponds to the gain between two frequencies
        # Now we have to repeat each row and column by the number of the fiber modes, so
        # that a block-like matrix is built. Each block contains the gain all the modes
        # at two frequencies
        #
        #            F1        F2
        #        +---------+---------+
        #        | m11 m12 | m11 m12 |
        #   F1   | m21 m22 | m21 m22 |
        #        +---------+---------+
        #        | m11 m12 | m11 m12 |
        #   F2   | m21 m22 | m21 m22 |
        #        +---------+---------+

        gain = gain.repeat_interleave(self.modes, dim=1).repeat_interleave(
            self.modes, dim=2
        )

        oi = self.overlap_integrals.expand((batch_size, self.modes, self.modes)).repeat(
            1, num_freqs, num_freqs
        )

        G = gain * oi
        # G = torch.zeros_like(G)

        solution = torch_rk4(
            MMFRamanAmplifier.ode, total_power, self.z, losses, G,
        ).view(-1, num_freqs, self.modes)

        signal_spectrum = solution[:, self.num_pumps :, :].clone()

        return signal_spectrum
