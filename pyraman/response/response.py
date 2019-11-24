import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate

from pyraman.utils import speed_of_light

positions = np.array([56.25, 100.00, 231.25, 362.50, 463.00, 497.00, 611.50,
                      691.67, 793.67, 835.50, 930.00, 1080.00, 1215.00]) * 1e2

intensities = np.array([1.00, 11.40, 36.67, 67.67, 74.00, 4.50, 6.80, 4.60, 4.20,
                        4.50, 2.70, 3.10, 3.00])

g_fwhm = np.array([52.10, 110.42, 175.00, 162.50, 135.33, 24.50, 41.50,
                   155.00, 59.50, 64.30, 150.00, 91.00, 160.00]) * 1e2

l_fwhm = np.array([17.37, 38.81, 58.33, 54.17, 45.11, 8.17, 13.83, 51.67,
                   19.83, 21.43, 50.00, 30.33, 53.33]) * 1e2

# Vibrational frequencies
omega_v = 2 * np.pi * speed_of_light * positions

# Lorentzian FWHM
gamma = np.pi * speed_of_light * l_fwhm

# Gaussian FWHM
Gamma = np.pi * speed_of_light * g_fwhm

# Amplitude
amplitudes = intensities * omega_v

# Number of vibrational components
num_components = len(positions)


def impulse_response(duration, fs, normalize=False, num_samples=None):
    """Compute the Raman impulse response of silica.

    Parameters
    ----------
    duration : float
        The duration of the impulse response in seconds.
    fs : float
        The sampling frequency.
    normalize : bool, optional
        Normalize to its maximum value, by default False
    num_samples : int, optional
        If specified, forces the number of samples, by default None

    Returns
    -------
    response : array_like
        The impulse response samples.
    t : array_like
        The time samples.
    """

    dt = 1 / fs

    if num_samples is None:
        num_samples = np.math.ceil(duration / dt)

    t = np.arange(num_samples) * dt

    modes = np.reshape(amplitudes / omega_v, (num_components, 1)) * \
        np.exp(-np.outer(gamma, t)) * np.exp(-np.outer(Gamma**2, t**2/4)) * \
        np.sin(np.outer(omega_v, t))

    response = np.sum(modes, axis=0)

    if normalize:
        response /= np.max(np.abs(response))

    return response, t


def gain_spectrum(frequencies, spacing=20e9, normalize=False, spline_order=4):
    """Compute the Raman gain spectrum of silica.

    Parameters
    ----------
    frequencies : array_like
        The frequency points at which the Raman gain must be computed.
    spacing : float, optional
        Desired frequency resolution in Hertz, by default 20e9
    normalize : bool, optional
        Normalize the gain profile to its maximum, by default False
    spline_order : int, optional
        The order of the spline used to interpolate the spectrum, by default 4

    Returns
    -------
    gains : array_like
        The interpolated gain values computed at frequencies.
    gain : array_like
        The entire Raman gain spectrum.
    f : array_like
        The frequency axis.
    """

    # Set the sampling frequency based on the maximum frequency requested
    fs = np.max(np.abs(frequencies)) * 10

    negative_idx = np.argwhere()

    # fs = max(fs, 80e12)

    num_samples = np.math.ceil(fs / spacing)

    # get the frequency response
    response, t = impulse_response(None, fs, num_samples=num_samples)

    # compute the frequency axis
    dt = 1 / fs
    duration = (num_samples - 1) * dt
    dF = fs / num_samples

    # ! This is not correct: at f = 0, gain is != 0
    # ! Maybe take the modulus of the negative frequencies and switch the sign
    # ! of the gain to force simmetry around f = 0
    f = (np.arange(num_samples)) * dF - (num_samples-1) * dF/2

    # obtain the Raman gain as the imaginary part of
    # the spectrum
    gain = -np.imag(np.fft.fft(response))
    gain = np.fft.fftshift(gain)

    if normalize:
        gain /= np.max(np.abs(gain))

    # interpolate at the desired frequency spacings
    spline = scipy.interpolate.splrep(f, gain, k=spline_order)
    gains = scipy.interpolate.splev(frequencies, spline)

    return gains, gain, f


if __name__ == "__main__":
    resp, t = impulse_response(1e-12, 1e14)

    freq = np.linspace(4e12, 40e12, 100)
    freq2 = -np.linspace(10e12, 40e12, 100)

    ff = np.stack((freq, freq2))

    raman_gain_interp, spectrum, f = gain_spectrum(ff, normalize=True)

    plt.clf()
    plt.plot(f * 1e-12, spectrum)
    plt.plot(ff.T * 1e-12, raman_gain_interp.T, marker='x')
    plt.show(block=False)
