import os
import pickle

def save_obj(obj, name, directory=''):
    if directory == '':
        directory = r"obj/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(directory+name+'.pkl', 'wb') as f:
        pickle.dump(obj, f)


def load_obj(name):
    with open('obj/'+name+'.pkl', 'rb') as f:
        return pickle.load(f)


def get_file_list(directory, ext='.wav'):
    return [fname for fname in os.listdir(directory) if fname.endswith(ext)]

