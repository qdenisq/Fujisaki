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
import utils
import subprocess
import os
import ctypes, sys
import win32com.shell.shell as shell

directory = 'C:\\Users\\s3628075\\Study\\Fujisaki\\DataBase\\'
autofuji_fname = r'C:\Users\s3628075\Study\Fujisaki\DataBase\Autofuji.exe'

wav_fnames = utils.get_file_list(directory)
for fname in wav_fnames:
    fuj_utils.convert_wav_to_f0_ascii(fname, directory)

print("wav_to_f0 completed")
print("f0_to_pac started")
f0_fnames = utils.get_file_list(directory, '.f0_ascii')
with open(directory+'dump.txt','w') as dumpfile:
    for fname in f0_fnames:
        thresh = 0.0001
        alpha = 2.0
        args ="{} 0 4 {} auto {}".format(directory+fname, thresh, alpha)
        subprocess.call(autofuji_fname+" "+args, stdout=dumpfile)
        print("{} f0_to_pac completed".format(fname))
print("f0_to_pac completed")


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

with open('m1nw0000pes_short.f0_ascii', 'wb') as f:
    for i in range(pitch.nframes):
        f0 = pitch.samp_values[i]
        vu = 1.0 if pitch.vuv[i] else 0.0
        fe = pitch.energy[i]*pitch.mean_energy
        line = '{} {} {} {}\n'.format(f0, vu, fe, vu)
        f.write(line)


output = np.log(pitch.values_interp)

# Find suitable parameters
params = {
          'Fb' : 74.610527,
          'a'  : 2.0,
          'b'  : 20.0,
          'y'  : 0.9,
          'I'  : 2,
          'Ap' : [0.4067, 0.3350],
          'T0p': [-0.38/t_end, 1.48/t_end],
          'J'  : 7,
          'Aa' : [0.6620, 0.3236, 0.2420, 0.3972, 0.3185, 0.3060, 0.3932],
          'T1a': [0.1726/t_end, 0.4275/t_end, 0.7000/t_end, 1.0325/t_end, 1.9500/t_end, 2.3902/t_end, 2.8950/t_end],
          'T2a': [0.3636/t_end, 0.6815/t_end, 1.0030/t_end, 1.1008/t_end, 2.1557/t_end, 2.8003/t_end, 3.0855/t_end]
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

import os
os.startfile("sample.wav")









plt.show()



