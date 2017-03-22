from __future__ import division
import fujisaki_model as fm
import numpy as np
import ml_utils
import amfm_decompy
import amfm_decompy.pYAAPT as pyaapt
import amfm_decompy.basic_tools as basic
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import fujisaki_utils as fuj_utils
import timeit


# Declare the variables.
file_name = "m1nw0000pes_short.wav"
# Create the signal object.
signal = basic.SignalObj(file_name)
# Get time interval and num_samples
t_start = 0.0
num_samples = signal.size
t_end = num_samples / signal.fs
t = np.linspace(t_start, t_end, num_samples)
# Create the pitch object and calculate its attributes.
pitch = pyaapt.yaapt(signal)
output = np.log(pitch.values_interp)

# Find suitable parameters
params = {
          'Fb' : 60.0,
          'a'  : 4.0,
          'b'  : 20.0,
          'y'  : 0.9,
          'I'  : 2,
          'Ap' : [0.55, 0.39, 0.5],
          'T0p': [0.03, 0.48, 0.6],
          'J'  : 11,
          'Aa' : [0.750, 0.30, 0.90, 0.45, 1.40, 0.20, 0.27, 0.50, 0.50, 4.900, 0.25],
          'T1a': [0.055, 0.14, 0.21, 0.23, 0.29, 0.35, 0.53, 0.64, 0.79, 0.901, 0.925],
          'T2a': [0.070, 0.20, 0.22, 0.28, 0.30, 0.42, 0.60, 0.79, 0.87, 0.903, 0.98]
          }

predictions = fuj_utils.generate_fujisaki_curve(show=False, verbose=False, time=t_end, fs=signal.fs, **params)
y = predictions['y']
Cp = predictions['Cp']
Ca = predictions['Ca']
Cp_sum = [sum(cp) for cp in zip(*Cp)] if len(Cp) != 0 else [0.0] * num_samples
Ca_sum = [sum(ca) for ca in zip(*Ca)] if len(Ca) != 0 else [0.0] * num_samples
# Plot
t_norm = np.linspace(0, 1, num_samples)
plt.plot(t_norm, y, label='guess')
plt.plot(t_norm, np.add(Cp_sum, predictions['y_b']), linestyle='--', label='phrase comp')
plt.plot(t_norm, np.add(Ca_sum, predictions['y_b']), linestyle='--', label='accent comp')

plt.plot(t_norm, np.log(pitch.values_interp), label='pich interpolation', color='green')
plt.xlabel('samples', fontsize=18)
plt.ylabel('pitch (Hz)', fontsize=18)
plt.legend(loc='upper right')









plt.show()



