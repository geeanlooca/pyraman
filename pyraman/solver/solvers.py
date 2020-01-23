import numpy as np
import scipy.integrate
import pyraman.utils
from pyraman.response import gain_spectrum, impulse_response
from pyraman.utils import wavelength_to_frequency, alpha_to_linear, watt_to_dBm, dBm_to_watt
import matplotlib.pyplot as plt
from pdb import set_trace


class Fiber:
    def __init__(self, core_diameter=8e-6, losses=0.2, raman_coefficient=7e-14,
                 effective_area=80e-12):
        
        self.effective_area = effective_area
        self.raman_coefficient = 7e-14        
        self.losses = alpha_to_linear(losses)
        self.losses_dB = losses
        self.raman_efficiency = self.raman_coefficient / self.effective_area
        self.raman_threshold = 16 * losses * self.raman_coefficient / self.effective_area
        
        
        super().__init__()
        
        
    def __str__(self):        
        description = f"""
        Raman coefficient: {self.raman_coefficient} m/W
        Raman efficiency: {self.raman_efficiency} 1/(m*W)
        Losses: {self.losses_dB} dB/km
        Losses: {self.losses:0.3} 1/m
        Raman threshold: {self.raman_threshold:0.3} W
        """        
        return description
        
    def __repr__(self):
        return self.__str__()


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

        return spline

    def _interpolate_gain(self, frequencies):

        # if negative frequencies are passed, just compute the gain
        # for the absolute value and then change the sign

        negative_idx = frequencies < 0

        pos_freqs = np.abs(frequencies)

        max_freq = np.max(pos_freqs)

        if max_freq > self.bandwidth:
            print("Warning: recomputing spline representation")
            self.bandwidth = 2 * max_freq

            # Recompute the gain spectrum with a bigger bandwidth
            self.spline = self._compute_gain_spectrum(self.bandwidth)

        # compute the interpolate values from the spline representation
        # we computed in the constructor
        gains = scipy.interpolate.splev(pos_freqs, self.spline)

        # switch the sign for negative frequencies
        gains[negative_idx] *= -1

        return gains

    def propagate(self):
        pass

    def solve(
        self,
        signal_power,
        signal_wavelength,
        pump_power,  
        pump_wavelength,
        z,
        losses=0.2,
        raman_coefficient=7e-14,
        effective_core_area=80e-12,
        check_photon_count=False,
    ):

        num_signals = signal_power.shape[0]
        num_pumps = pump_power.shape[0]
        
        total_signals = num_pumps + num_signals
        
        wavelengths = np.concatenate((pump_wavelength, signal_wavelength))        
        input_power = np.concatenate((pump_power, signal_power))

        frequencies = wavelength_to_frequency(wavelengths)        
        
        losses_linear = alpha_to_linear(losses)

        # Compute the frequency shifts for each signal
        frequency_shifts = np.zeros((total_signals, total_signals))
        for i in range(total_signals):
            frequency_shifts[i, :] = frequencies - frequencies[i]

        gains = self._interpolate_gain(frequency_shifts)

        gains *= raman_coefficient / effective_core_area
        
        raman_threshold = 16 * losses_linear *  raman_coefficient / effective_core_area

        # Force diagonal to be 0
        np.fill_diagonal(gains, 0)
        # gains = np.triu(gains) + np.triu(gains, 1).T
        
        # compute the frequency scaling factor
        freqs = np.expand_dims(frequencies, axis=-1)
        freq_scaling = np.maximum(1, freqs * (1 / freqs.T))

        # gain matrix
        gain_matrix = freq_scaling * gains
        

        sol = scipy.integrate.odeint(
            RamanAmplifier.raman_ode, input_power, z, args=(losses_linear, gain_matrix)
        )
        
        pump_solution = sol[:, :num_pumps]
        signal_solution = sol[:, num_pumps:]

        if check_photon_count:
            photon_count = np.sum(sol / frequencies, axis=1)
            return pump_solution, signal_solution, photon_count
        else:
            return pump_solution, signal_solution

    @staticmethod
    def raman_ode(P, z, losses, gain_matrix):
        dPdz = (-losses + np.matmul(gain_matrix, P[:, np.newaxis])) * P[:, np.newaxis]
        return np.squeeze(dPdz)


if __name__ == "__main__":
    ampl = RamanAmplifier(bandwidth=40e12)

    freqs = np.linspace(-30, 30, 1000) * 1e12
    gains = ampl._interpolate_gain(freqs)    
     
    
    losses = 0.2
    fiber = Fiber(losses=losses)
    
    pump_wavelength = np.array([1430]) * 1e-9
    pump_power = np.array([100 * 1e-3])
    num_pumps = pump_power.shape[0]
    
    band = pyraman.utils.bands.C + pyraman.utils.bands.L
    num_channels = 1
    input_power = 0
    power_per_channel = dBm_to_watt(input_power) / num_channels
    
    signal_wavelength = np.linspace(*band , num_channels)
    
    signal_power =  power_per_channel * np.ones_like(signal_wavelength)        
    total_power = watt_to_dBm(np.sum(signal_power))
    
    print(f"Total launch power: {total_power} dBm")

    powers = np.concatenate((signal_power, pump_power))
    wavelengths = np.concatenate((signal_wavelength, pump_wavelength))
    
    length = 50 * 1e3

    z = np.linspace(0, length, 1000)

    pump_sol, signal_sol = ampl.solve(signal_power, signal_wavelength,
                     pump_power, pump_wavelength, z, losses=fiber.losses_dB, 
                     raman_coefficient=fiber.raman_coefficient,
                     effective_core_area=fiber.effective_area)
    
    _, sol_off = ampl.solve(signal_power, signal_wavelength,
                     0 * pump_power, pump_wavelength, z, losses=fiber.losses_dB, 
                     raman_coefficient=fiber.raman_coefficient,
                     effective_core_area=fiber.effective_area)
    
    onoff_gain = signal_sol[-1] / sol_off[-1]

    output_spectrum = signal_sol[-1]
    
    

    fig = plt.figure(1)
    plt.clf()
    fig, axs = plt.subplots(ncols=3, nrows=1, num=1)

    axs[0].plot(z * 1e-3, watt_to_dBm(pump_sol))
    axs[0].plot(z * 1e-3, watt_to_dBm(signal_sol))
    axs[0].plot(z * 1e-3, watt_to_dBm(sol_off), '--')
    axs[0].set_xlabel('Position [km]')  
    axs[0].grid(which='both')
    
    
    stem_bottom = watt_to_dBm(np.min(output_spectrum)) - 5
    
    axs[1].stem(signal_wavelength * 1e9, watt_to_dBm(output_spectrum), bottom=stem_bottom)
    axs[1].stem(signal_wavelength * 1e9, watt_to_dBm(sol_off[-1]), bottom=stem_bottom, linefmt='r')
    axs[1].set_xlabel('Wavelength [\lambda]')
    axs[1].grid(which='both')
    axs[1].set_title('Output spectrum [dBm]')
    
    stem_bottom = 10*np.log10(np.min(onoff_gain)) - 5

    axs[2].stem(signal_wavelength * 1e9, 10*np.log10(onoff_gain), bottom=stem_bottom)
    axs[2].set_xlabel('Wavelength [\lambda]')
    axs[2].set_title('On-off gain [dB]')
    axs[2].grid(which='both')
    
    

    # plt.plot(z, 10*np.log10(sol[:, 2]))
    
    plt.show(block=False)
