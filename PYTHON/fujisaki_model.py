import numpy as np

"""
Create Fujisaki object
"""


class FujisakiObj(object):
    def __init__(self, Fb=1.0, a=3.0, b=20.0, y=0.9, Ap=[], T0p=[], T2a=[], T1a=[], Aa=[], **kwargs):
        self.Fb = Fb
        self.a = a
        self.b = b
        self.y = y

        self.Ap = Ap
        self.T0p = T0p

        if (len(self.Ap) != len(self.T0p)):
            print ('vectors of Phrase Commands and their timings have different length\n'
                   'Ap : {}; T0p : {}'.format(len(self.Ap), len(self.T0p)))

        self.Aa = Aa
        self.T1a = T1a
        self.T2a = T2a

        if (len(self.Aa) != len(self.T1a) or len(self.Aa) != len(self.T2a)):
            print ('vectors of Accent Commands and their timings have different length\n'
                   'Aa : {}; T1a : {}; T2a : {}'.format(len(self.Aa), len(self.T1a), len(self.T2a)))

    def __repr__(self, ):
        return "Fb={}; a={}; b={}; y={}".format(self.Fb, self.a, self.b, self.y)


def calc_Gp(a, t):
    res = a ** 2 * t * np.exp(-a * t)
    for i in range(len(res)):
        if res[i] < 0:
            res[i] = 0
    return res

def calc_Ga(b, y, t_input):
    res = [min(y, 1.0 - (1.0 + b * t) * np.exp(-b * t)) for t in t_input]
    for i in range(len(res)):
        if t_input[i] < 0:
            res[i] = 0
    return res


"""
calc phrase component in time t: sum(Ap*Gp(t-T0p)
"""
def calc_phrase_component(fuj_obj, t):
    res = 0.0
    for i in range(len(fuj_obj.Ap)):
        res += fuj_obj.Ap[i] * calc_Gp(fuj_obj.a, t - fuj_obj.T0p[i])
    return res


"""
calc accent component in time t: sum(Aa*(Ga(t-T1a) - G1(t-T2a)
"""
def calc_accent_component(fuj_obj, t):
    res = 0.0
    b = fuj_obj.b
    y = fuj_obj.y
    for i in range(len(fuj_obj.Aa)):
        res += fuj_obj.Aa[i] * (calc_Ga(b, y, t - fuj_obj.T1a[i]) - calc_Ga(b, y, t - fuj_obj.T2a[i]))
    return res


"""
predict fujisaki f0 given model parameters in time window (t_start, t_end))
"""
def predict_fujisaki(fuj_obj, t_start, t_end, num_samples):
    t = np.linspace(start=t_start, stop=t_end, num=num_samples)
    ln_f0 = [np.log(fuj_obj.Fb) + calc_accent_component(fuj_obj, ti) + calc_phrase_component(fuj_obj, ti) for ti in t]
    return ln_f0



def fujisaki_func(input, *params):
    # fb, a, b, y, I, J, _ = params
    params = list(params)
    fb = params[0]
    a = params[1]
    b = params[2]
    y = params[3]
    I = params[4]
    J = params[5]
    Ap = []
    T0p = []
    for i in range(6, 2*int(I)+6, 2):
        Ap.append(params[i])
        T0p.append(params[i+1])

    Aa = []
    T1a = []
    T2a = []
    for i in range(2 * int(I) + 6, 3 * int(J) + 2 * int(I) + 6, 3):
        Aa.append(params[i])
        T1a.append(params[i + 1])
        T2a.append(params[i + 2])

    def phrase_comp(Api, a, T0pi, t) :
        return Api*calc_Gp(a, t-T0pi)

    def accent_comp(Aai, b, y, T1ai, T2ai, t):
        return Aai*(np.subtract(calc_Ga(b, y, t-T1ai),calc_Ga(b, y, t-T2ai)))
    ln_f0 = np.log(fb) + sum([phrase_comp(Ap[i], a, T0p[i], input) for i in range(len(Ap))]) \
        + sum([accent_comp(Aa[j], b, y, T1a[j], T2a[j], input) for j in range(len(Aa))])

    return ln_f0
