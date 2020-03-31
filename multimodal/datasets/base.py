
from __future__ import print_function
import numpy as np
import numpy.ma as ma
from multimodal.datasets.data_sample import DataSample

from six.moves import cPickle as pickle #for performance

import numpy as np


def load_npz_X_y(filename_):
    with np.load(filename_) as npzfile:
       return npzfile['X'] , npzfile['y']

def save_npz_X_y(filename_, X, y):
    np.savez(filename_, X=X, y=y)

def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di


def _create_pickle_files(adr, dsample):
    f = open(adr + ".sample.pkl", "wb")
    pickle.dump(dsample, f)
    f.close()
