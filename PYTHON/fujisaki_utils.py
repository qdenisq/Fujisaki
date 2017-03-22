import numpy as np
import fujisaki_model as fm
import matplotlib.pyplot as plt
import pickle
import os
def generate_fujisaki_params(min_Fb = 20, max_Fb = 500, min_a = 0.0, max_a = 10.0,min_b = 0.0, max_b = 40.0,min_I = 1, max_I = 10, min_J = 1, max_J = 10, verbose = True):

    Fb = np.random.random()*max_Fb
    a = min_a + np.random.random()*(max_a-min_a)
    b = min_b + np.random.random()*(max_b-min_b)
    y = np.random.random()

    # Phrase command number
    I = np.random.randint(min_I, max_I)
    Ap = [0.5]*I - np.random.rand(I)
    T0p = np.random.rand(I)
    T0p.sort()

    # Accent command number
    J = np.random.randint(min_J, max_J)
    Aa = np.random.rand(J)/2
    t = np.random.rand(2*J)
    t.sort()
    T1a = t[0:2*J:2]
    T2a = t[1:2*J:2]

    # Scale command timings with signal length
    # num_samples = time/fs
    # T0p = T0p*num_samples
    # T1a = T1a*num_samples
    # T2a = T2a*num_samples

    if verbose:
        print "Fujisaki parameters:\n" \
              "Fb = {}\n" \
              "a = {}\n" \
              "b = {}\n" \
              "y = {}\n" \
              "I = {}\n" \
              "Ap = {}\n" \
              "T0p = {}\n" \
              "J = {}\n" \
              "Aa = {}\n" \
              "T1a = {}\n" \
              "T2a = {}".format(Fb, a, b, y, I, Ap, T0p, J, Aa, T1a, T2a)
    # Create dict and return it
    p = {}
    for i in ('Fb', 'a', 'b', 'y', 'I', 'J', 'Ap', 'T0p', 'Aa', 'T1a', 'T2a'):
        p[i] = locals()[i]
    return p


def generate_fujisaki_curve(**kwargs):
    # Parse params
    p = kwargs
    show = p.get('show', True)
    verbose = p.get('verbose', False)
    time = p.get('time', 5.0)
    fs = p.get('fs', 20000)

    Fb = p['Fb']
    a = p['a']
    b = p['b']
    y = p['y']
    I = p['I']
    J = p['J']
    Ap = p['Ap']
    T0p = p['T0p']
    Aa = p['Aa']
    T1a = p['T1a']
    T2a = p['T2a']

    # Create time array
    num_samples = int(time*fs)
    x = np.linspace(0, time, num_samples)
    # Scale timings
    T0p = [ti*time for ti in T0p]
    T1a = [ti*time for ti in T1a]
    T2a = [ti*time for ti in T2a]

    # Base frequency component
    y_b = [np.log(Fb)]*num_samples
    # Phrase command components
    Cp = [Ap[i]*fm.calc_Gp(a, x - T0p[i]) for i in range(I)]
    # Accent command components
    Ca = [Aa[j]*(np.subtract(fm.calc_Ga(b, y, x - T1a[j]), fm.calc_Ga(b, y, x - T2a[j]))) for j in range(J)]

    Cp_sum = [sum(cp) for cp in zip(*Cp)] if I != 0 else [0.0]*num_samples
    Ca_sum = [sum(ca) for ca in zip(*Ca)] if J != 0 else [0.0]*num_samples

    y = [sum(comp) for comp in zip(y_b, Ca_sum, Cp_sum)]

    if verbose == True:
        print sum(Cp_sum)
        print len(Cp_sum)
        print sum(Ca_sum)
        print len(Ca_sum)

    if show == True:
        plt.plot(x, y, linewidth=2.0, label='output')
        plt.plot(x, y_b, linestyle='--', label='base comp')
        plt.plot(x, Cp_sum, linestyle='--', label='phrase comp')
        plt.plot(x, Ca_sum, linestyle='--', label='accent comp')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()
    return {'x': x,
            'y': y,
            'Ca': Ca,
            'Cp': Cp,
            'y_b': y_b}


def save_obj(obj, name):
    dir = r"obj/"
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(dir+name+'.pkl', 'wb') as f:
        pickle.dump(obj, f)


def load_obj(name):
    with open('obj/'+name+'.pkl', 'rb') as f:
        return pickle.load(f)

