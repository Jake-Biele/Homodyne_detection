
# coding: utf-8

# ## Experimental AC Data Plotter
# 
# Author: Jake Biele  
# Data: 25/01/22  
# Email: jakebiele7@gmail.com
# 
# Note: you will need to install the relevant depedencies including pandas, scipy, os and numpy. The figures are made using a home built plotting module titled fig (saved in the containing folder).
# 
# Here we plot data imported from the folder AC data to make Figs. 1(c), 2(a)-2(b).


import pandas as pd
import importlib
import fig
importlib.reload(fig)
import numpy as np
from scipy.optimize import curve_fit
import os
import scipy.constants as cst
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

# Import DC data
path = os.getcwd()
data_folder = '/Data/AC data/Vac shot noise data/'
spectrum_data_file = 'spectrums.csv'
power_data_file = 'powerdata.csv'

# Import data from csv
s_data = pd.read_csv(path+data_folder+spectrum_data_file, header = 1) 
p_data = pd.read_csv(path+data_folder+power_data_file, header = 2)

# Experimental parameter
f = cst.e/((cst.h*cst.c)/2070e-9) # - Central frequency (m)
RBW = 100000 # - ESA resolution bandwidth (Hz)
gain = 3914/2 # - Effective feedback gain (Ohms)
res = 1.66 # - Photodiode responsivity (A/W)


# Useful functions

# Butterworth gain spectrum
def GG(f, fstar, p, A):
    
    return A/(1+(p**2-2)*(f/fstar)**2 + (f/fstar)**4)

# Gaussian kernel moving point average
def movingaverage_gauss(x_values, y_values, FWHM):
    output = []
    
    sigma = FWHM/np.sqrt(8 * np.log(2))
    y_values = 10**(np.array(y_values)/10)
    
    for i in range(len(y_values)):
        
        kernel_at_pos = np.exp(-(np.array(x_values) - x_values[i]) ** 2 / (2 * sigma ** 2))
        kernel_at_pos = kernel_at_pos/sum(kernel_at_pos)
        
        ma = np.sum(y_values*kernel_at_pos)
        output.append(10*np.log10(ma))
        
    return output


# ## PSD with power plot

# Extract specific data for plottig - linear steps in dBm:
powers = p_data.iloc[[1,2,4,6,8,10,12,14],0]
frequency, spectra = s_data.iloc[0:,0].tolist(), np.transpose(np.array(s_data.iloc[0:,[1,2,4,6,8,10,12,14]])).tolist()

# Convert powers to linear and scale by tap power splitting ratio
powers = [10*np.log10(10**(float(x)/10)*(78.9/9.57)) for x in powers]

# Scale frequency to MHz
frequency = [float(f)/1e6 for f in frequency]

# Build legend for the plot
leg = []
leg = [r'${:3.1f}$ dBm'.format(p) for p  in powers]
leg.pop(0)
leg.insert(0, 'Background')

# Insert frequency into spectra array for plotting as a single array with first column taken as x
spectra.insert(0, frequency)

# Plot PSD 
plot = fig.fig(spectra)
    
plot.x_label = 'Frequency [MHz]'
plot.y_label = 'PSD [dBm]'

plot.y_range = [-98, -84]
plot.x_range = [2, 24]

plot.markers = ['--', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-']
plot.figsize = (10, 6.25*(3/3))
plot.joined = False

plot.legend = leg
plot.bbox = (1.1,-0.2)
plot.legend_ncols = 3

plot.plot()
# plot.plot(target='file',filename=path+'1bleg.pdf')


# ## Linear variance with power plot

# Plot linearity of noise taking average around 3MHz: 1-19 MHz

# Extract data:
spectra, powers = np.transpose(np.array(s_data.iloc[20:380,1:-4])).tolist(), p_data.iloc[0:-4,0]

# Convert powers to linear and scale by tap power splitting ratio
logPower = [10*np.log10(10**(float(x)/10)*(78.9/9.57)) for x in powers]

# Average PSD over bandwidth - converting to linear to take average
aver = []

for each in range(np.shape(spectra)[0]):
    aver.append(np.sum(10**(np.array(spectra[each])/10))/len(spectra[each]))

# Subtract the detector noise (aver[0])
aver = [x for x in aver]
shift = [x-aver[0] for x in aver]

# Convert both x and y to log scale
logVar = [10*np.log10(x) for x in aver]
logShift = [10*np.log10(x) for x in shift]

# Cut saturation data out for accurate linear plotting
x = logPower[1:-3]
y = logShift[1:-3]

# Fit data to linear
fit = np.polyfit(x,y,1)
print(fit)
fit_fn = np.poly1d(fit)

# Use scipy to extract linear fit parameters
from scipy import stats
slope, intercept, r_value, p_value, std_err = stats.linregress([[i,j] for i,j in zip(x,y)])
print("slope = ", slope)
print("intercept = ", intercept)
print("R = ", r_value)
print("p = ", p_value)
print("Standard error = ", std_err)
print(slope+2*std_err)

# Create noise floor array for plotting
nf = [10*np.log10(aver[0])]*len(logPower)

# Plot data on log scales with noise floor, noise corrected and raw signals (including saturation points)
plot = fig.fig([[logPower[1:], logVar[1:]],
            [logPower[1:], logShift[1:]],
            [logPower[1:-2], fit_fn(logPower[1:-2])],
            [logPower[1:], nf[1:]]])
    
plot.x_label = 'Power [dBm]'
plot.y_label = 'Log Variance ' + r'$[V^2]$'

plot.markers = ['o', '^', '-', '-']
plot.colours = ['#00035b', '#e50000', '#e50000','#9a0eea']

plot.legend = ['Raw signal variance', 'Noise subtracted signal', 'Linear fit ' + r'$(m=0.99 \pm 0.02)$', 'Detector noise floor']
plot.legend_loc = 'lower right'
plot.figsize = (10, 10)
plot.bbox = (2.1,-0.2)
plot.joined = False

# plot.plot(target='file',filename=path+'1cleg.pdf')
plot.plot()


# ## Gain spectrum fitting

# Import DC data
path = os.getcwd()
data_folder = '/Data/AC data/Gain spectrum data/'
spectrum_data_file = 'spectrums.csv'
power_data_file = 'powerdata.csv'

# Import data from csv
s_data = pd.read_csv(path+data_folder+spectrum_data_file, header = 1) 
p_data = pd.read_csv(path+data_folder+power_data_file, header = 1)


from scipy import constants as cst

# Extract data
frequency = s_data.iloc[0:,0].tolist()
frequency = [float(f)/1e6 for f in frequency]
spectra, powers = np.transpose(np.array(s_data.iloc[0:,[1,18]])).tolist(), p_data.iloc[18,0]
Power = 10**(float(powers)/10)*(78.9/9.57) 

# Convert to liner and correct for background noise
spectra = [10**(np.array(s)/10)-10**(np.array(spectra[0])/10) for s in spectra]
spectra.pop(0)


# Fit data to a second-order butterworth over 1-24 MHz
popt, _ = curve_fit(GG, frequency[1:frequency.index(24)], spectra[0][1:480])
print(popt)

# Calculate residuals
sig = spectra[0][1:480]
residuals = sig - GG(frequency[1:frequency.index(24)], *popt)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((sig-np.mean(sig))**2)
r_squared = 1 - (ss_res / ss_tot)
print(r_squared)

# Make the plot
plot = fig.fig([[frequency,10*np.log10(spectra[-1])+90.2],
                 [frequency,10*np.log10(GG(frequency, popt[0], popt[1], popt[2]))+90.2],
                 [frequency,[-3]*len(frequency)]])
    
plot.x_label = 'Frequency [MHz]'
plot.y_label = r'Normalised Gain [dB]'
plot.y_range = [-10, 1]

plot.logx = True
plot.x_range = [1, 25]

plot.markers = ['o', '-','--']
plot.markersize = 4

plot.legend = ['Data', 'Model Fit','3-dB']
plot.legend_loc = 3

plot.joined = False
plot.plot()
plot.plot(target='file',filename=path+'/fig4c.pdf')

