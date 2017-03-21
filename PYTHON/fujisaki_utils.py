import numpy as np
import fujisaki_model as fm
import matplotlib.pyplot as plt

def generate_fujisaki_params(max_Fb = 500, max_a = 3.0, max_b = 20.0, max_I = 10, max_J = 10, verbose = True):

    Fb = np.random.random()*max_Fb
    a = np.random.random()*max_a
    b = np.random.random()*max_b
    y = np.random.random()

    # Phrase command number
    I = np.random.randint(0, max_I)
    Ap = [0.5]*I - np.random.rand(I)
    T0p = np.random.rand(I)
    T0p.sort()

    # Accent command number
    J = np.random.randint(0, max_J)
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


def generate_fujisaki_curve(show = True, verbose = False, time = 5.0, fs = 20000, **kwargs):
    # Create time array
    num_samples = int(time*fs)
    x = np.linspace(0, time, num_samples)
    # Parse params
    p = kwargs
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
    # Scale timings
    T0p = T0p*time
    T1a = T1a*time
    T2a = T2a*time

    # Base frequency component
    y_b = [np.log(Fb)]*num_samples
    # Phrase command components
    Cp = [Ap[i]*fm.calc_Gp(a, x - T0p[i]) for i in range(I)]

    # Accent command components
    Ca = [Aa[j]*(np.subtract(fm.calc_Ga(b, y, x - T1a[j]), fm.calc_Ga(b, y, x - T2a[j]))) for j in range(J)]

    Cp_sum = [sum(cp) for cp in zip(*Cp)] if I != 0 else [0.0]*num_samples
    Ca_sum = [sum(ca) for ca in zip(*Ca)] if J != 0 else [0.0]*num_samples

    if verbose:
        print sum(Cp_sum)
        print len(Cp_sum)
        print sum(Ca_sum)
        print len(Ca_sum)

    y = [sum(comp) for comp in zip(y_b, Ca_sum, Cp_sum)]

    if show == True:
        plt.plot(x, y, linewidth=2.0, label='output')
        plt.plot(x, Cp_sum, linestyle='--', label='phrase comp')
        plt.plot(x, Ca_sum, linestyle='--', label='accent comp')
        plt.legend()
        plt.show()
    return x, y


