import amfm_decompy.pYAAPT as pyaapt
import amfm_decompy.basic_tools as basic
import numpy as np
import fujisaki_model as fm
import matplotlib.pyplot as plt
import utils
import shlex
import subprocess, os, sys, getopt
import subprocess32
import multiprocessing
from joblib import Parallel, delayed
import time
import pandas

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
    show = p.get('show', False)
    verbose = p.get('verbose', False)
    x = p['t']

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

    num_samples = len(x)
    # Base frequency component
    y_b = [np.log(Fb)]*len(x)
    # Phrase command components
    Cp = [Ap[i]*fm.calc_Gp(a, x - T0p[i]) for i in range(I)]
    # Accent command components
    Ca = [Aa[j]*(np.subtract(fm.calc_Ga(b, y, x - T1a[j]), fm.calc_Ga(b, y, x - T2a[j]))) for j in range(J)]

    Cp_sum = [sum(cp) for cp in zip(*Cp)] if I != 0 else [0.0]*num_samples
    Ca_sum = [sum(ca) for ca in zip(*Ca)] if J != 0 else [0.0]*num_samples

    output = [sum(comp) for comp in zip(y_b, Ca_sum, Cp_sum)]

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
            'output': output,
            'Ca': Ca,
            'Cp': Cp,
            'y_b': y_b}


def calc_fuj_error(f0, f0_interp, x, fuj_params):
    num_samples = len(f0_interp)
    fuj = generate_fujisaki_curve(t=x, y=0.9, **fuj_params)
    f0_fuj = fuj['output']
    # rss = np.square(np.subtract(f0, f0_fuj))
    # mean_rss = sum(rss)/float(num_samples)
    rss_interp = np.square(np.subtract(f0_interp, f0_fuj))
    mean_rss_interp = sum(rss_interp)/float(num_samples)
    return {'rss_interp': rss_interp,
            'mean_rss_interp': mean_rss_interp}


def analyze(fuj_params, rmse, verbose=False):
    subjects = {}
    emotions = {}
    i = 0
    em_keys = ['sa', 'su', 'an', 'fe', 'di', 'ha']
    s_keys = np.linspace(1, 44, 44)
    subj_keys = ['s'+str(s) for s in s_keys]
    for key, value in rmse.iteritems():
        i += 1
        if verbose:
            print "{} of {}: {} is being analyzed ".format(i, len(rmse), key)
        # Parse filename and find emotion category, subject
        try:
            s = ''
            for subj in subj_keys:
                if subj in key:
                    s = subj
            e = ''
            for em in em_keys:
                if em in key:
                    e = em
        except ValueError as e:
            print key, e.message
            continue
        if s != '':
            if s not in subjects:
                subjects[s] = []
            subjects[s].append(value)
        if e != '':
            if e not in emotions:
                emotions[e] = []
            emotions[e].append(value)
    return subjects, emotions


def convert_avi_to_wav(fname, directory=''):
    print 'convert ', fname
    command = "ffmpeg.exe -i \"{}\" -ab 160k -ac 2 -ar 44100 -vn \"{}\"".format(fname, os.path.splitext(fname)[0]+".wav")
    subprocess.call(command, shell=True)


def convert_wav_to_f0_ascii(fname, fs, directory=''):
    print 'convert ', fname
    # Create the signal object.
    signal = basic.SignalObj(fname)
    # Get time interval and num_samples
    t_start = 0.0
    num_samples = signal.size
    t_end = num_samples / signal.fs
    t = np.linspace(t_start, t_end, num_samples)
    # Create the pitch object and calculate its attributes.
    pitch = pyaapt.yaapt(signal)
    # Create .f0_ascii file and dump f0 to it
    output_fname = directory+os.path.splitext(os.path.basename(fname))[0]+'.f0_ascii'
    with open(output_fname, 'wb') as f:
        for i in range(pitch.nframes):
            f0 = pitch.samp_values[i]
            vu = 1.0 if pitch.vuv[i] else 0.0
            fe = pitch.energy[i] * pitch.mean_energy
            line = '{} {} {} {}\n'.format(f0, vu, fe, vu)
            f.write(line)
    step = signal.fs / fs
    output_f0 = pitch.values_interp[0:signal.size:step]
    return (os.path.splitext(os.path.basename(fname))[0], output_f0)


def convert_f0_ascii_to_pac(fname, autofuji_fname, directory=''):
    if sys.platform.startswith('win'):
        import ctypes
        SEM_NOGPFAULTERRORBOX = 0x0002
        # ctypes.windll.kernel32.SetErrorMode(SEM_NOGPFAULTERRORBOX)
        import win32con

        subprocess_flags = win32con.CREATE_NO_WINDOW
    else:
        subprocess_flags = 0

    print 'convert ', fname
    thresh = 0.0001
    min_delta = 100.0
    best_alpha = 1.0
    # Call autofuji.exe for every alpha and check for the min delta
    for alpha in np.linspace(1.0, 3.0, 20):
        args = "\"{}\" 0 4 {} auto {}".format(directory + fname, thresh, alpha)
        list_args = shlex.split(autofuji_fname + " " + args)
        output = ''
        try:
            output = subprocess32.check_output(list_args, timeout=30.0)
            delta_str = output.splitlines()[-1]
            delta = float(delta_str.split()[-1])
        except Exception as e:
            if os.path.exists(os.path.splitext(directory + fname)[0]+'.PAC'):
                os.remove(os.path.splitext(directory + fname)[0]+'.PAC')
            print e.message
            return
        if delta < min_delta:
            min_delta = delta
            best_alpha = alpha
    args = "\"{}\" 0 4 {} auto {}".format(directory + fname, thresh, best_alpha)
    try:
        subprocess32.call(autofuji_fname + " " + args, shell=True, stdout=open(os.devnull, 'w'), timeout=5.0)
    except Exception as e:
        if os.path.exists(os.path.splitext(directory + fname)[0] + '.PAC'):
           pass# os.remove(os.path.splitext(directory + fname)[0] + '.PAC')


def parse_pac_file(fname):
    with open(fname, 'r') as f:
        try:
            lines = f.readlines()
            I = int(lines[7])
            J = int(lines[8])
            Fb = float(lines[9])
            # Parse phrase components
            T0p = []
            Ap = []
            a = 0.0
            for i in range(20, 20+I):
                t0p, _, ap, a = lines[i].split()
                T0p.append(float(t0p))
                Ap.append(float(ap))

            # Parse accent components
            T1a = []
            T2a = []
            Aa = []
            b = 0.0
            for i in range(20 + I, 20 + I + J):
                t1a, t2a, aa, b = lines[i].split()
                T1a.append(float(t1a))
                T2a.append(float(t2a))
                Aa.append(float(aa))

            return {'Fb': Fb, 'a': float(a), 'b': float(b), 'I': I, 'J': J, 'Ap': Ap, 'T0p': T0p, 'Aa': Aa, 'T1a': T1a, 'T2a': T2a}
        except Exception as e:
            print e
            return


def transform_params(fuj_params, phrase_count=3, accent_count=12):
    type = [('name', 'S10'), ('Fb', 'f'), ('a', 'f')]
    for i in range(phrase_count):
        type.append(('Ap{}'.format(i), 'f'))
    for i in range(phrase_count):
        type.append(('T0p{}'.format(i), 'f'))
    for i in range(accent_count):
        type.append(('Aa{}'.format(i), 'f'))
    for i in range(accent_count):
        type.append(('T1a{}'.format(i), 'f'))
    for i in range(accent_count):
        type.append(('T2a{}'.format(i), 'f'))

    x = np.empty(len(fuj_params), dtype=np.dtype(type))
    i = 0
    for name, params in fuj_params.items():
        # labels = 'name', 'Fb', 'a', 'Ap0', 'Ap1', 'Ap2', 'T0p0'..., '
        record = [name]
        fb = params['Fb']
        a = params['a']

        record.append(fb)
        record.append(a)

        Ap_list = params['Ap']
        Ap_list.extend([None]*phrase_count)
        Ap_list = Ap_list[:phrase_count]
        record.extend(Ap_list)

        T0p_list = params['T0p']
        T0p_list.extend([None] * phrase_count)
        T0p_list = T0p_list[:phrase_count]
        record.extend(T0p_list)

        Aa_list = params['Aa']
        Aa_list.extend([None]*accent_count)
        Aa_list = Aa_list[:accent_count]
        record.extend(Aa_list)

        T1a_list = params['T1a']
        T1a_list.extend([None]*accent_count)
        T1a_list = T1a_list[:accent_count]
        record.extend(T1a_list)

        T2a_list = params['T2a']
        T2a_list.extend([None]*accent_count)
        T2a_list = T2a_list[:accent_count]
        record.extend(T2a_list)
        x[i] = tuple(record)
        i += 1
    return x


def main(argv):

    # directory = r'D:\Emotional Databases\IEMOCAP\IEMOCAP_full_release\Session1\sentences\wav\Ses01F_script01_1/'
    # directory = r'C:\Users\s3628075\Study\Fujisaki\DataBase\BerlinDatabase\All'
    # directory = r'C:\Users\s3628075\Study\Fujisaki\DataBase\PolishEmotionalSpeechDataAGH\EmotiveKorpus\All'
    directory = r'C:\Users\s3628075\Study\Fujisaki\DataBase\enterface\All/'
    # directory = r'C:/Users/s3628075/Study/Fujisaki/DataBase/Ses01F_script01_1'
    autofuji_fname = r'C:/Users/s3628075/Study/Fujisaki_estimator/AutoFuji.exe'
    key = 0
    fs = 1500

    try:
        opts, args = getopt.getopt(argv,"hd:k:",["directory=","key="])
    except getopt.GetoptError:
        print 'fujisaki_utils.py -d <directory> -k <key>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'fujisaki_utils.py -d <directory> -k <key>\n' \
                  '-d <directory>: directory to look up\n' \
                  '-k <key>: operation to perform\n' \
                  '0 - all\n' \
                  '1 - wav to f0_ascii\n' \
                  '2 - f0_ascii to pac'
            sys.exit()
        elif opt in ("-d", "--directory"):
            directory = arg
        elif opt in ("-k", "--key"):
            key = int(arg)
        elif opt == '-fs':
            fs = arg
    directory = directory.replace('\\', '/')
    if not directory.endswith('/'):
        directory += '/'
    print 'Working directory is ', directory
    print 'Key is', key

    start = time.time()
    if key == 0 or key == 1:
        # print '//////////////////////////////////////////////////////////\n' \
        #       'Convert avi files to wav in', directory
        # avi_fnames = utils.get_file_list(directory, '.avi')
        # Parallel(n_jobs=multiprocessing.cpu_count(), backend='threading', verbose=5)(
        #     delayed(convert_avi_to_wav)(directory + fname, directory) for fname in avi_fnames)
        # # for fname in wav_fnames:
        # #     print 'convert ', fname
        # #     convert_wav_to_f0_ascii(directory+fname, directory)
        # print 'Conversion avi files to wav finished'

        print '//////////////////////////////////////////////////////////\n' \
              'Convert wav files to f0_ascii in', directory
        wav_fnames = utils.get_file_list(directory, '.wav')
        results = Parallel(n_jobs=multiprocessing.cpu_count(), backend='threading', verbose=5)(
            delayed(convert_wav_to_f0_ascii)(directory+fname, fs, directory) for fname in wav_fnames)
        f0_i = dict(results)
        utils.save_obj(f0_i, 'F0_signal', directory)
        # for fname in wav_fnames:
        #     print 'convert ', fname
        #     convert_wav_to_f0_ascii(directory+fname, directory)
        print 'Conversion wav files to f0_ascii finished. Elapsed: ', time.time()-start

    if key == 0 or key == 2:
        print '//////////////////////////////////////////////////////////\n' \
              'Convert f0_ascii to PAC in', directory

        f0_fnames = utils.get_file_list(directory, '.f0_ascii')
        Parallel(n_jobs=multiprocessing.cpu_count(), backend='threading', verbose=20)(
            delayed(convert_f0_ascii_to_pac)(fname, autofuji_fname, directory) for fname in f0_fnames)
        # for fname in f0_fnames:
        #     print 'convert ', fname
            # convert_f0_ascii_to_pac(fname, autofuji_fname, directory)
        print 'Conversion f0_ascii to PAC finished. Elapsed: ', time.time()-start

    if key == 0 or key == 3:
        print '/////////////////////////////////////////////////////////\n' \
              'Create report for all .Pac files in ', directory
        pac_fnames = utils.get_file_list(directory, '.PAC')
        p_all = {}
        with open(directory+'Report.rep', 'w') as f:
            i = 0
            f0_i = utils.load_obj(directory+'F0_signal.pkl')
            f0_fuj = {}
            rmse = {}
            for fname in pac_fnames:
                i += 1
                print '{} of {} : convert {}'.format(i, len(pac_fnames), fname)
                params = parse_pac_file(directory+fname)
                if params == None:
                    continue
                f.write('{} {}\n'.format(fname, params))
                name_key = os.path.splitext(fname)[0]
                f0_interp = np.log(f0_i[name_key])
                t_end = len(f0_interp)/fs
                x = np.linspace(start=0.0, stop=t_end, num=len(f0_interp))
                f0_f = generate_fujisaki_curve(t=x, y=0.9, **params)
                f0_fuj[name_key] = f0_f
                rss_interp = np.square(np.subtract(f0_interp, f0_f['output']))
                rmse[name_key] = sum(rss_interp) / float(len(f0_interp))
                p_all[name_key] = params
        print 'Saving Fujisaki_Params.pkl in ', directory
        utils.save_obj(p_all, 'Fujisaki_Params', directory)
        print 'Saving Fujisaki_Contour.pkl in ', directory
        utils.save_obj(f0_fuj, 'Fujisaki_Contour', directory)
        print 'Saving RMSE.pkl in ', directory
        utils.save_obj(rmse, 'RMSE', directory)
        print 'Report creating finished. Elapsed: ', time.time()-start

    if key == 0 or key == 4:
        print '/////////////////////////////////////////////////////////\n' \
              'Analyze report in ', directory
        fuj_params = utils.load_obj(directory+'Fujisaki_Params.pkl')
        rmse = utils.load_obj(directory+'RMSE.pkl')
        subj, emot = analyze(rmse=rmse, fuj_params=fuj_params, verbose=True)
        utils.save_obj(subj, 'Subjects', directory)
        utils.save_obj(emot, 'Emotions', directory)
        print 'Analysis finished. Elapsed: ', time.time()-start
    print '//////////////////////////////////////////////////////////\n' \
          '/////////////////////// Finish ///////////////////////////\n' \
          '//////////////////////////////////////////////////////////'
if __name__ == "__main__":
   main(sys.argv[1:])