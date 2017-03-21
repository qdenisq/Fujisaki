import numpy as np


def calc_rss(output, prediction):
    return sum((output - prediction)**2)
