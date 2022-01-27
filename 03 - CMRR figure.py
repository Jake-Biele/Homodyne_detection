
# coding: utf-8

# ## Experimental CMRR Data Plotter
# 
# Author: Jake Biele  
# Data: 25/01/22  
# Email: jakebiele7@gmail.com
# 
# Note: you will need to install the relevant depedencies including pandas, scipy, os and numpy. The figures are made using a home built plotting module titled fig (saved in the containing folder).
# 
# Here we plot analyse CMRR data with a plot that was not included in the final manuscript.

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
data_file = '/Data/CMRR data/CMRR.csv'
data = pd.read_csv(path+data_file, header = 0) 
data.head(5)

# Extract data
frequency, spectra = data.iloc[0:-1,0].tolist(), np.transpose(np.array(data.iloc[0:-1,[1,2]])).tolist()
frequency = [float(x)/1e6 for x in frequency]
spectra.insert(0,frequency[0:])

# Extract CMRR
CMRR = np.max(spectra[1][frequency.index(30):frequency.index(45)])-np.max(spectra[2][frequency.index(30):frequency.index(45)]) - 6
print('CMRR = {:.0f} dB'.format(CMRR))


# Plot data
plot = fig.fig(spectra)
    
plot.x_label = 'Frequency [MHz]'
plot.y_label = 'Power Spectral Density [dBm]'
plot.x_range = [30, 50]
plot.legend = ['Addition','Subtraction']
plot.joined = True
plot.plot()
# plot.plot(target='file',filename=path+'CMRR_repRate.pdf')

