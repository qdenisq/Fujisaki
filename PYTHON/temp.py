from __future__ import division
import fujisaki_model as fm
import numpy as np
import ml_utils
import amfm_decompy
import amfm_decompy.pYAAPT as pyaapt
import amfm_decompy.basic_tools as basic
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import timeit

# Declare the variables.
file_name = "m1nw0000pes_short.wav"
window_duration = 0.015   # in seconds
nharm_max = 25
SNR = float('Inf')



# Create the signal object.
signal = basic.SignalObj(file_name)

# Get time interval and num_samples
t_start = 0.0
num_samples = signal.size
t_end = num_samples / signal.fs
t = np.linspace(t_start, t_end, num_samples)
# Create the pitch object and calculate its attributes.
pyaapt.PitchObj.SMOOTH_FACTOR = 5
pyaapt.PitchObj.SMOOTH = 5

pitch = pyaapt.yaapt(signal)
output = np.log(pitch.values_interp + 1.0)

# Plot pitch values
plt.plot(output, label='pich interpolation', color='green')


I = 1
J = 5
guess = [50.0, 3.0, 20.0, 0.9, I, J]

print guess
guess += np.random.random_sample((I,)).tolist()
guess += (num_samples*np.random.random_sample((I,))).tolist()
guess += np.random.random_sample((J,)).tolist()
t1a = (num_samples*np.random.random_sample((J,))).tolist()
guess += t1a
guess += (np.add(t1a, np.absolute((100*np.random.random_sample((J,))).tolist()))).tolist()

print(guess)
start = timeit.timeit()
popt, pcov = curve_fit(fm.fujisaki_func, t, output, p0=guess)
end = timeit.timeit()
predict = fm.fujisaki_func(t, *popt)
plt.plot(predict)
print(popt)
print ("elapsed time : {}".format(end-start))


# # Prepare features set
# parameters = {}
# parameters['Fb'] = np.linspace(start=10.0, stop=600.0, num=100)
# parameters['a'] = np.linspace(0.0, 2.0, num=10)
# parameters['y'] = np.linspace(0.0, 2.0, num=10)
# parameters['b'] = np.linspace(0.0, 2.0, num=10)
#
# fuj_obj_list = [fm.FujisakiObj(Fb = fb, a = a, b = b, y = 0.9,  Aa=[1.0], T1a=[0.6], T2a=[2.0])
#                 for fb in parameters['Fb']
#                 for a in parameters['a']
#                 for b in parameters['b']
#                ]
#
# min_rss = 1e5
# best_fuj = None
# res = []
# for fuj in fuj_obj_list:
#     rss = ml_utils.calc_rss(output, fm.predict_fujisaki(fuj, t_start, t_end, num_samples))
#     res.append((fuj, rss))
#     print("Fujisaki params: {}; RSS: {}".format(fuj, rss))
#     if rss < min_rss:
#         best_fuj = fuj
#         min_rss = rss
#
#
# # Predict fujisaki object
# predictions = fm.predict_fujisaki(best_fuj, t_start, t_end, num_samples)
#
# # Plot pitch prediction
# plt.plot(predictions, label='predictions', color='blue')
#
#


plt.show()
