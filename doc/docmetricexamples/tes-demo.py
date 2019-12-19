import numpy as np
from sklearn import datasets
from sklearn.metrics.pairwise import rbf_kernel
from metriclearning.mvml import MVML
from metriclearning.lpMKL import MKL
from metriclearning.datasets.data_sample import DataSample
from metriclearning.tests.datasets.get_dataset_path import get_dataset_path
import pickle
np.random.seed(4)

# =========== create a simple dataset ============

n_tot = 200
half = int(n_tot/2)
n_tr = 120

# create a bit more data than needed so that we can take "half" amount of samples for each class
X0, y0 = datasets.make_moons(n_samples=n_tot+2, noise=0.3, shuffle=False)
X1, y1 = datasets.make_circles(n_samples=n_tot+2, noise=0.1, shuffle=False)

# make multi-view correspondence (select equal number of samples for both classes and order the data same way
# in both views)

yinds0 = np.append(np.where(y0 == 0)[0][0:half], np.where(y0 == 1)[0][0:half])
yinds1 = np.append(np.where(y1 == 0)[0][0:half], np.where(y1 == 1)[0][0:half])

X0 = X0[yinds0, :]
X1 = X1[yinds1, :]
Y = np.append(np.zeros(half)-1, np.ones(half))  # labels -1 and 1

n_tot = 200
half = int(n_tot/2)
n_tr = 120

# create a bit more data than needed so that we can take "half" amount of samples for each class
X0, y0 = datasets.make_moons(n_samples=n_tot+2, noise=0.3, shuffle=False)
X1, y1 = datasets.make_circles(n_samples=n_tot+2, noise=0.1, shuffle=False)

# make multi-view correspondence (select equal number of samples for both classes and order the data same way
# in both views)

yinds0 = np.append(np.where(y0 == 0)[0][0:half], np.where(y0 == 1)[0][0:half])
yinds1 = np.append(np.where(y1 == 0)[0][0:half], np.where(y1 == 1)[0][0:half])

X0 = X0[yinds0, :]
X1 = X1[yinds1, :]
Y = np.append(np.zeros(half)-1, np.ones(half))  # labels -1 and 1


# shuffle
order = np.random.permutation(n_tot)
X0 = X0[order, :]
X1 = X1[order, :]
Y = Y[order]

# make kernel dictionaries
kernel_dict = {}
test_kernel_dict = {}
kernel_dict[0] = rbf_kernel(X0[0:n_tr, :])
kernel_dict[1] = rbf_kernel(X1[0:n_tr, :])
test_kernel_dict[0] = rbf_kernel(X0[n_tr:n_tot, :], X0[0:n_tr, :])
test_kernel_dict[1] = rbf_kernel(X1[n_tr:n_tot, :], X1[0:n_tr, :])

d= DataSample(kernel_dict)
a = d.data
# np.save(input_x, kernel_dict)
# np.save(input_y, Y)
# f = open(input_x, "wb")
# pickle.dump(input_x, f)
#input_x = get_dataset_path("input_x_dic.pkl")
#f = open(input_x, "r")
#dicoc = pickle.load(f)
# pickle.dump(kernel_dict, f)
#f.close()
# =========== use MVML in classifying the data ============

# demo on how the code is intended to be used; parameters are not cross-validated, just picked some
# mvml = MVML(kernel_dict, Y[0:n_tr], [0.1, 1], nystrom_param=0.2)


mvml = MVML( [0.1, 1], nystrom_param=0.2)
mvml.fit(a, Y[0:n_tr])
print("x shape", mvml.X_.shape)
print("x shape int",mvml.X_.shapes_int)
dd = DataSample(test_kernel_dict)
X_test = dd.data
red1 = mvml.predict(X_test)

mkl = MKL(lmbda=0.1)
mkl.fit(kernel_dict,Y[0:n_tr] )

mkl.predict(X_test)
#red1 = np.sign(mvml.predict_mvml(test_kernel_dict, g1, w1))




