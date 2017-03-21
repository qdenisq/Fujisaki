#!/usr/bin/env python

"""
Script to test the AMFM_decompy package.

Version 1.0.6
23/Jan/2017 Bernardo J.B. Schmitt - bernardo.jb.schmitt@gmail.com
"""
import amfm_decompy
import amfm_decompy.pYAAPT as pyaapt
import amfm_decompy.pyQHM as pyqhm
import amfm_decompy.basic_tools as basic
import os.path
from matplotlib import pyplot as plt

# Declare the variables.
file_name = "C:\\Users\\s3628075\\Study\\Fujisaki\\f0_detectors_lib\\AMFM_decompy-1.0.6\\amfm_decompy\\m1nw0000pes_short.wav"
window_duration = 0.015   # in seconds
nharm_max = 25
SNR = float('Inf')

# Create the signal object.
signal = basic.SignalObj(file_name)

# Create the window object.
#window = pyqhm.SampleWindow(window_duration, signal.fs)

# Create the pitch object and calculate its attributes.
pitch = pyaapt.yaapt(signal)

plt.plot(pitch.values, label='pchip interpolation', color='green')
plt.show()
#
# # Set the number of modulated components.
# signal.set_nharm(pitch.values, nharm_max)
#
# # Check if gaussian noise has to be added.
# if SNR != float('Inf'):
#     signal.noiser(pitch.values, SNR)
#
# # Perform the QHM extraction.
# QHM = pyqhm.qhm(signal, pitch, window, 0.001, N_iter = 3, phase_tech = 'phase')
#
# print ("QHM SRER: {}".format(QHM.SRER))
# #
# # Perform the aQHM extraction.
# aQHM = pyqhm.aqhm(signal, QHM, pitch, window, 0.001, N_iter = 3, N_runs = 2,
#             phase_tech = 'phase')
#
# print ("aQHM SRER: {}".format(aQHM.SRER))
#
# # Perform the eaQHM extraction.
# eaQHM = pyqhm.eaqhm(signal, aQHM, pitch, window, 0.001, N_iter=3, N_runs=2,
#               phase_tech = 'phase')
#
# print ("eaQHM SRER: {}".format(eaQHM.SRER))
#
