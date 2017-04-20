import numpy as np


def calc_rss(output, prediction):
    return sum((output - prediction)**2)

    params=[]
    for i in range(1):
        params.extend(stat_utlis.get_significant_parameters(directory, verbose=True))
    params = np.unique(params)
    print params

    data = utils.load_obj(directory+'fuj_params_data.pkl')
    # filter out sample with nan in params
    names = data['name']



def rmfield(a, *fields):
    return a[[name for name in a.dtype.names if name not in fields]]
