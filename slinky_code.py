#Q5 and Q6
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

#model functions for curve_fit
def one_exp_model(x, a):
    return (0.025/((np.exp(a*1.87))-np.exp(-a*1.87)))*np.exp(a*x)

def two_exp_model(x, a):
    return (0.025/(np.exp(a*1.87)-np.exp(-a*1.87)))*(np.exp(a*x)-np.exp(-a*x))

def reduced_chisquared(data_y, model_y, data_error, dof):
    return (1/(data_y.size-dof))*np.sum(((data_y-model_y)/data_error)**2)

#loading min and max position data
data_driven_0 = np.loadtxt(".\Desktop\physics\PHY224-slinky\data\data_driven_0.txt", skiprows = 1)
data_driven_10 = np.loadtxt(".\Desktop\physics\PHY224-slinky\data\data_driven_10.txt", skiprows = 1)
data_driven_40 = np.loadtxt(".\Desktop\physics\PHY224-slinky\data\data_driven_40.txt", skiprows = 1)

#creating arrays to put equilibrium position and amplitude data inside
data_amplitude_0 = np.zeros((np.size(data_driven_0[:,0]),2))
data_amplitude_10 = np.zeros((np.size(data_driven_10[:,0]),2))
data_amplitude_40 = np.zeros((np.size(data_driven_40[:,0]),2))

#inserting position data
data_amplitude_0[:,0] = (data_driven_0[:,1] + data_driven_0[:,0])/2
data_amplitude_10[:,0] = (data_driven_10[:,1] + data_driven_10[:,0])/2
data_amplitude_40[:,0] = (data_driven_10[:,1] + data_driven_10[:,0])/2

#inserting amplitude data
data_amplitude_0[:,1] = (data_driven_0[:,1] - data_driven_0[:,0])/2
data_amplitude_10[:,1] = (data_driven_10[:,1] - data_driven_10[:,0])/2
data_amplitude_40[:,1] = (data_driven_10[:,1] - data_driven_10[:,0])/2

#equilibrium position error
error_position_0 = np.full(np.size(data_driven_0[:,0]), np.sqrt(0.005**2 + 0.005**2)/2)
error_position_10 = np.full(np.size(data_driven_10[:,0]), np.sqrt(0.005**2 + 0.005**2)/2)
error_position_40 = np.full(np.size(data_driven_40[:,0]), np.sqrt(0.005**2 + 0.005**2)/2)

#amplitude error
error_amplitude_0 = np.full(np.size(data_driven_0[:,0]), np.sqrt(0.005**2 + 0.005**2)/2)
error_amplitude_10 = np.full(np.size(data_driven_10[:,0]), np.sqrt(0.005**2 + 0.005**2)/2)
error_amplitude_40 = np.full(np.size(data_driven_40[:,0]), np.sqrt(0.005**2 + 0.005**2)/2)

#curve_fit for optimized parameter value and variance for one exponential
popt1_0, pcov1_0 = curve_fit(one_exp_model, data_amplitude_0[:,0], 
                         data_amplitude_0[:,1], p0=[2.1797], 
                         sigma=error_amplitude_0, absolute_sigma=True)
popt1_10, pcov1_10 = curve_fit(one_exp_model, data_amplitude_10[:,0], 
                         data_amplitude_10[:,1], p0=[2.1279], 
                         sigma=error_amplitude_10, absolute_sigma=True)

#curve_fit for optimized parameter value and variance for two exponential
popt2_0, pcov2_0 = curve_fit(two_exp_model, data_amplitude_0[:,0], 
                         data_amplitude_0[:,1], p0=[2.1797], 
                         sigma=error_amplitude_0, absolute_sigma=True)
popt2_10, pcov2_10 = curve_fit(two_exp_model, data_amplitude_10[:,0], 
                         data_amplitude_10[:,1], p0=[2.1279], 
                         sigma=error_amplitude_10, absolute_sigma=True)

#obtaining y value from optimized parameters for one and two exponential
model_one_exp_0 = one_exp_model(data_amplitude_0[:,0], popt1_0)
model_one_exp_10 = one_exp_model(data_amplitude_10[:,0], popt1_10)
model_two_exp_0 = two_exp_model(data_amplitude_0[:,0], popt2_0)
model_two_exp_10 = two_exp_model(data_amplitude_10[:,0], popt2_10)

#plotting the graphs for q5
plt.figure(0)
plt.errorbar(data_amplitude_0[:,0], data_amplitude_0[:,1], xerr=error_position_0,
             yerr=error_amplitude_0, fmt='.', label='Data')
plt.plot(data_amplitude_0[:,0], model_one_exp_0, '-', label='One Exponential Model')
plt.plot(data_amplitude_0[:,0], model_two_exp_0, '-', label='Two Exponential Model')
plt.title("Position from Fixed End vs Amplitude with omega approaching 0")
plt.xlabel("Position from Fixed End (m)")
plt.ylabel("Amplitude (m)")
plt.legend()

plt.figure(1)
plt.errorbar(data_amplitude_10[:,0], data_amplitude_10[:,1], xerr=error_position_10,
             yerr=error_amplitude_10, fmt='.', label='Data')
plt.plot(data_amplitude_10[:,0], model_one_exp_10, '-', label='One Exponential Model')
plt.plot(data_amplitude_10[:,0], model_two_exp_10, '-', label='Two Exponential Model')
plt.title("Position from Fixed End vs Amplitude with Motor Driven at 10%")
plt.xlabel("Position from Fixed End (m)")
plt.ylabel("Amplitude (m)")
plt.legend()

#calculating reduced chi squared
chisq_red1_0 = reduced_chisquared(data_amplitude_0[:,1], model_one_exp_0, 
                                  error_amplitude_0,  1)
chisq_red2_0 = reduced_chisquared(data_amplitude_0[:,1], model_two_exp_0, 
                                  error_amplitude_0,  1)
chisq_red1_10 = reduced_chisquared(data_amplitude_10[:,1], model_one_exp_10, 
                                  error_amplitude_10,  1)
chisq_red2_10 = reduced_chisquared(data_amplitude_10[:,1], model_two_exp_10, 
                                  error_amplitude_10,  1)
print(chisq_red1_0)
print(chisq_red2_0)
print(chisq_red1_10)
print(chisq_red2_10)