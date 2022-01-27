
# coding: utf-8

# ## Experimental DC Data Plotter
# 
# Author: Jake Biele  
# Data: 25/01/22  
# Email: jakebiele7@gmail.com
# 
# Note: you will need to install the relevant depedencies including pandas, scipy, os and numpy. The figures are made using a home built plotting module titled fig (saved in the containing folder).
# 
# Here we plot data imported from DC data.csv to make Fig. 1(b).


import pandas as pd
import importlib
import fig
importlib.reload(fig)
import numpy as np
from scipy.optimize import curve_fit
import os
import scipy.constants as cst
import warnings
warnings.filterwarnings('ignore')

# Import DC data
path = os.getcwd()
data_file = '/Data/DC data.csv'
data = pd.read_csv(path+data_file, header = 1) 
data.head(5)

# Define experimental parameters - for more metadata see AC data/metadata.csv.

split = 9.57/78.9 # - power tap splotting ratio
R = 3870 # - feedback resistor (Ohms)
wave = 2070e-9 # - central wavelength of the pulsed laser (m)
scope_input_r = 1e6 # - input impedance of the oscillascope (Ohms)
sr = 51 # - detector output serier resistance (Ohms)

# Useful functions
def eta(m):
    
    # Potential divider between detector and scope
    pot = scope_input_r/(scope_input_r+sr)
    
    # Laser frequency
    f = cst.e/((cst.h*cst.c)/wave)
    
    # Convert units and return eta as 0 to 1
    return (m/(R*split*pot*1e-3*f))*100

def linfunc(x,m):
    return x*m


# Extract PD+ data
x = data.iloc[0:,0].tolist()
xerr = data.iloc[0:,1].tolist()
y = data.iloc[0:,2].tolist()
yerr = data.iloc[0:,3].tolist()

# Extract PD- data
x2 = data.iloc[0:,4].tolist()
xerr2 = data.iloc[0:,5].tolist()
y2 = data.iloc[0:,6].tolist()
yerr2 = data.iloc[0:,7].tolist()

# Fit data to linear trend
popt, _ = curve_fit(linfunc, x, y)
eta1 = eta(popt)
popt, _ = curve_fit(linfunc, x2, y2)
eta2 = eta(popt)

# Extrapolate trend for plotting
lin1 = [linfunc(i,popt[0]) for i in x]
lin1err = [0,0,0,0,0,0]

# Scale x data by power tap splitting ratio
x = [i*split for i in x]
x2 = [i*split for i in x2]
xerr = [i*split for i in xerr]
xerr2 = [i*split for i in xerr2]

# Plot data
plot = fig.fig([[x,y],[x2,y2]])

plot.x_label = r'Incident Power [$mW$]'
plot.y_label = 'DC Voltage [V]'

plot.markers = ['o','o','-','--']
plot.colours = ['#00035b', '#e50000', '#00035b','#e50000','#e50000']
plot.joined = False

plot.xerr = [xerr, xerr2]
plot.yerr = [yerr, yerr2]

plot.legend = [r'$PD^+$', r'$PD^-$','']

# plot.regression = True

plot.annotate = [r'$\eta_{tot} =$' + ' {:.3f} '.format(eta1[0]/100) + r'$\pm 0.015$',
                 r'$\eta_{tot} =$' + ' {:.3f} '.format(np.abs(eta2[0]/100)) + r'$\pm 0.020$']

plot.annotate_xys = [(0.75,0.67),(0.75,0.33)]
plot.annotate_size = 24

# plot.plot(target = 'file', filename = path + 'fig4b.pdf')
plot.plot()

