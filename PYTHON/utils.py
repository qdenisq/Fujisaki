import os
import pickle
from matplotlib import pyplot as plt
import numpy as np


def save_obj(obj, name, directory=''):
    if directory == '':
        directory = r"obj/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(directory+name+'.pkl', 'wb') as f:
        pickle.dump(obj, f)


def load_obj(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


def get_file_list(directory, ext='.wav'):
    return [fname for fname in os.listdir(directory) if fname.endswith(ext)]


def plot_hist(a, title, binwidth, xlabel='', verbose=False):
    plt.figure()
    h = plt.hist(a, bins=np.arange(min(a), max(a) + binwidth, binwidth), normed=True)
    plt.title(title)
    if xlabel == '':
        xlabel = title
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    if verbose:
        print "{} length: {}; " \
              "min_value: {}; " \
              "max_value: {}".format(title, len(a), min(a), max(a))
    return h
