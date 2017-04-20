import scipy.stats as stats
import utils
import matplotlib.pyplot as plt
import numpy as np
import random
import fujisaki_utils
from pprint import pprint


def check_normality(data, ind_var=[], dep_var='', verbose=True):
    d = {}
    assert dep_var != ''
    for k, v in data.items():

        for var in ind_var:
            if var in k:
                if var not in d.keys():
                    d[var] = []
                try:
                    d[var].extend(v[dep_var])
                except:
                    d[var].append(v[dep_var])
    res = {}
    for var in ind_var:
        v = d[var]
        v = random.sample(v, 40)
        # v = np.log(v-np.min(v)+1)

        args = stats.lognorm.fit(v)
        res[var] = stats.kstest(v, 'lognorm', args)
        if verbose:
            print 'normality test var=\'{}\'  {} : {}'.format(dep_var, var, res[var])
            plt.hist(v)
            plt.show()
    return res


def calc_anova(directory, thresh = 0.05, verbose=True):
    data = utils.load_obj(directory+'fuj_params_data.pkl')
    ind_var = ['sa', 'su', 'an', 'fe', 'di', 'ha']
    anova_test = {}
    for p in data.dtype.names:
        if p == 'name':
            continue
        p_values = []
        distributions=[]
        for e in ind_var:
            # Get data from this emotion e
            e_data = np.array([data[i] for i in range(len(data)) if e in data[i]['name']], dtype=data.dtype)
            # Check normality
            v = e_data[p]
            v = [x for x in v if not np.isnan(x)]
            distributions.append(v)
            try:
                v = random.sample(v, 30)
            except ValueError as err:
                if verbose == True:
                    print '{} {} sample size:{}. {}'.format(p, e, len(v), err.message)
                continue
            stat, p_val = stats.normaltest(v)
            p_values.append(p_val)
            if verbose == True:
                print '{} {} {}'.format(p, e, p_val)
        thresh = 0.05
        # Decide parametric or non-parametric ANOVA to use
        if len(p_values) < 2:
            continue

        if sum(i > thresh for i in p_values) > len(p_values)/2:
            # Use parametric ANOVA because more than half of distributions are normal
            _, anova_test[p] = stats.f_oneway(*distributions)
        else:
            _, anova_test[p] = stats.kruskal(*distributions)
    if verbose == True:
        print 'ANOVA TEST RESULT:'
        pprint(anova_test)
    return anova_test



