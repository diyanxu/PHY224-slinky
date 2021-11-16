#Q5 and Q6
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

#model functions for curve_fit
def one_exp_model(x, a, b):
    return b*np.exp(a*x)

def two_exp_model(x, a, b):
    return b*(np.exp(a*x)-np.exp(-a*x))

def linear_model(x, a):
    return a*x

def reduced_chisquared(data_y, model_y, data_error, dof):
    return (1/(data_y.size-dof))*np.sum(((data_y-model_y)/data_error)**2)

def q3_model(x, a, b):
    return a * x + b

# Reading data
data_driven_0 = np.loadtxt("data/data_driven_0.txt", skiprows = 1)
data_driven_10 = np.loadtxt("data/data_driven_10.txt", skiprows = 1)
data_driven_40 = np.loadtxt("data/data_driven_40.txt", skiprows = 1)
data_nodes = np.loadtxt("data/data_node.txt", skiprows=1)


#creating arrays to put equilibrium position and amplitude data inside
data_amplitude_0 = np.zeros((np.size(data_driven_0[:,0]),2))
data_amplitude_10 = np.zeros((np.size(data_driven_10[:,0]),2))
data_amplitude_40 = np.zeros((np.size(data_driven_40[:,0]),2))

#inserting position data
data_amplitude_0[:,0] = (data_driven_0[:,1] + data_driven_0[:,0])/2
data_amplitude_10[:,0] = (data_driven_10[:,1] + data_driven_10[:,0])/2
data_amplitude_40[:,0] = (data_driven_40[:,1] + data_driven_40[:,0])/2

#inserting amplitude data
data_amplitude_0[:,1] = (data_driven_0[:,1] - data_driven_0[:,0])/2
data_amplitude_10[:,1] = (data_driven_10[:,1] - data_driven_10[:,0])/2
data_amplitude_40[:,1] = (data_driven_40[:,1] - data_driven_40[:,0])/2

#equilibrium position error
error_position_0 = np.full(np.size(data_driven_0[:,0]),
                           np.sqrt(0.005**2 + 0.005**2)/2)
error_position_10 = np.full(np.size(data_driven_10[:,0]),
                            np.sqrt(0.005**2 + 0.005**2)/2)
error_position_40 = np.full(np.size(data_driven_40[:,0]),
                            np.sqrt(0.005**2 + 0.005**2)/2)

#amplitude error
error_amplitude_0 = np.full(np.size(data_driven_0[:,0]),
                            np.sqrt(0.005**2 + 0.005**2)/2)
error_amplitude_10 = np.full(np.size(data_driven_10[:,0]),
                             np.sqrt(0.005**2 + 0.005**2)/2)
error_amplitude_40 = np.full(np.size(data_driven_40[:,0]),
                             np.sqrt(0.005**2 + 0.005**2)/2)

#curve_fit for optimized parameter value and variance for one exponential
popt1_0, pcov1_0 = curve_fit(one_exp_model, data_amplitude_0[:,0], 
                         data_amplitude_0[:,1], p0=[2.1797, 0.00042449], 
                         sigma=error_amplitude_0, absolute_sigma=True)
popt1_10, pcov1_10 = curve_fit(one_exp_model, data_amplitude_10[:,0], 
                         data_amplitude_10[:,1], p0=[2.1279, 0.00046769], 
                         sigma=error_amplitude_10, absolute_sigma=True)

#curve_fit for optimized parameter value and variance for two exponential
popt2_0, pcov2_0 = curve_fit(two_exp_model, data_amplitude_0[:,0], 
                         data_amplitude_0[:,1], p0=[2.1797, 0.00042449], 
                         sigma=error_amplitude_0, absolute_sigma=True)
popt2_10, pcov2_10 = curve_fit(two_exp_model, data_amplitude_10[:,0], 
                         data_amplitude_10[:,1], p0=[2.1279, 0.00046769], 
                         sigma=error_amplitude_10, absolute_sigma=True)

#curve_fit for optimized parameter value and variance for linear
popt_linear, pcov_linear = curve_fit(linear_model, data_amplitude_40[:,0],
                               data_amplitude_40[:,1], p0=[0.013369],
                               sigma=error_amplitude_40, absolute_sigma=True)

#obtaining y value from optimized parameters for one and two exponential and 
#linear
model_one_exp_0 = one_exp_model(data_amplitude_0[:,0], popt1_0[0], 
                                popt1_0[1])
model_one_exp_10 = one_exp_model(data_amplitude_10[:,0], popt1_10[0], 
                                 popt1_10[1])
model_two_exp_0 = two_exp_model(data_amplitude_0[:,0], popt2_0[0], 
                                popt2_0[1])
model_two_exp_10 = two_exp_model(data_amplitude_10[:,0], popt2_10[0], 
                                 popt2_10[1])
model_linear = linear_model(data_amplitude_40[:,0], popt_linear)

#plotting the graphs for q5 and q6
plt.figure(0)
plt.errorbar(data_amplitude_0[:,0], data_amplitude_0[:,1],
             xerr=error_position_0, yerr=error_amplitude_0, fmt='.',
             label='Data')
plt.plot(data_amplitude_0[:,0], model_one_exp_0, '-',
         label='One Exponential Model')
plt.plot(data_amplitude_0[:,0], model_two_exp_0, '-',
         label='Two Exponential Model')
plt.title("Position from Fixed End vs Amplitude with ω approaching 0")
plt.xlabel("Position from Fixed End (m)")
plt.ylabel("Amplitude (m)")
plt.legend()

plt.figure(1)
plt.errorbar(data_amplitude_10[:,0], data_amplitude_10[:,1],
             xerr=error_position_10, yerr=error_amplitude_10, fmt='.',
             label='Data')
plt.plot(data_amplitude_10[:,0], model_one_exp_10, '-',
         label='One Exponential Model')
plt.plot(data_amplitude_10[:,0], model_two_exp_10, '-',
         label='Two Exponential Model')
# fixed strings to follow PEP8
plt.title("Position from Fixed End vs Amplitude with Motor Driven at 10%" +
          "(ω << ω0)")
plt.xlabel("Position from Fixed End (m)")
plt.ylabel("Amplitude (m)")
plt.legend()

plt.figure(2)
plt.errorbar(data_amplitude_40[:,0], data_amplitude_40[:,1],
             xerr=error_position_40, yerr=error_amplitude_40, fmt='.',
             label='Data')
plt.plot(data_amplitude_40[:,0], model_linear, '-', label='Linear Model')
plt.title("Position from Fixed End vs Amplitude with Motor Driven at 40%" +
          "(ω = ω0)")
plt.xlabel("Position from Fixed End (m)")
plt.ylabel("Amplitude (m)")
plt.legend()

#calculating reduced chi squared
chisq_red1_0 = reduced_chisquared(data_amplitude_0[:,1], model_one_exp_0, 
                                  error_amplitude_0,  2)
chisq_red2_0 = reduced_chisquared(data_amplitude_0[:,1], model_two_exp_0, 
                                  error_amplitude_0,  2)
chisq_red1_10 = reduced_chisquared(data_amplitude_10[:,1], model_one_exp_10, 
                                  error_amplitude_10,  2)
chisq_red2_10 = reduced_chisquared(data_amplitude_10[:,1], model_two_exp_10, 
                                  error_amplitude_10,  2)
chisq_red_linear = reduced_chisquared(data_amplitude_40[:,1], model_linear, 
                                  error_amplitude_40,  1)
print(chisq_red1_0)
print(chisq_red2_0)
print(chisq_red1_10)
print(chisq_red2_10)
print(chisq_red_linear)

# code for experiment 3
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
wave_error_prec = data_node_length_error/data_node_length
data_node_wave_error = data_node_wave * wave_error_prec

#TODO: fix error calc here
# squaring data, doubling error
data_node_freq = data_node_freq**2
data_node_freq_error = data_node_freq*2*freq_error_prec
data_node_wave = data_node_wave**2
data_node_wave_error = data_node_wave*2*wave_error_prec

popt_q3, pcov_q3 = curve_fit(q3_model, data_node_wave, 
                       data_node_freq, p0=[2.359, 11.209], 
                       sigma=data_node_freq_error, absolute_sigma=True)

plt.figure(3)
plt.errorbar(data_node_wave, data_node_freq, yerr=data_node_freq_error,
             fmt='.', label='data')
plt.plot(data_node_wave, q3_model(data_node_wave, *popt_q3), label='model')
plt.title("Angular frequency squared vs. Wavenumber squared")
plt.legend()

print(popt_q3)

plt.show()