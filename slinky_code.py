#Q5 and Q6
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

#model functions for curve_fit
def one_exp_model(x, a):
    return (0.025/((np.exp(a*1.87))-np.exp(-a*1.87)))*np.exp(a*x)

def two_exp_model(x, a):
    return (0.025/(np.exp(a*1.87)-np.exp(-a*1.87)))*(np.exp(a*x)-np.exp(-a*x))

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
error_position = np.sqrt(0.005**2 + 0.005**2)/2

#amplitude error
error_amplitude = np.sqrt(0.005**2 + 0.005**2)/2

#curve_fit for optimized parameter value and variance
popt1, pcov1 = curve_fit(one_exp_model, data_amplitude_0[:,0], 
                         data_amplitude_0[:,1], p0=[2.1797], 
                         sigma=error_position, absolute_sigma=True)