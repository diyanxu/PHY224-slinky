import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# a is c_0, b is omega_0
def q3_model(x, a, b):
    return a * x + b


data_nodes = np.loadtxt("data/data_node.txt", skiprows=1)
data_node_time = data_nodes[:,0]
data_node_time_error = data_nodes[:,1]
data_node_length = data_nodes[:,2]
data_node_length_error = data_nodes[:,3]
# angular frequency is 2pi/time
data_node_freq = (1/data_node_time) * (2*np.pi)
# error for frequency
freq_error_prec = data_node_time_error/data_node_time
data_node_freq_error = data_node_freq * freq_error_prec

# getting wave number as 2pi/wavelength
data_node_wave = (1/data_node_length) * (2*np.pi)
print(data_node_wave)
wave_error_prec = data_node_length_error/data_node_length
data_node_wave_error = data_node_wave * wave_error_prec

# squaring data, doubling error
data_node_freq = data_node_freq**2
data_node_freq_error = data_node_freq_error*2
data_node_wave = data_node_wave**2
data_node_wave_error = data_node_wave_error*2

popt, pcov = curve_fit(q3_model, data_node_wave, 
                       data_node_freq, p0=[2.359, 11.209], 
                       sigma=data_node_freq_error, absolute_sigma=True)

print(popt)

plt.figure(0)
plt.errorbar(data_node_freq, data_node_wave, yerr=data_node_length_error)
