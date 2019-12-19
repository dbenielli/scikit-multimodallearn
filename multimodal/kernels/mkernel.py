import numpy as np
import scipy as sp
from sklearn.metrics.pairwise import pairwise_kernels
from abc import ABCMeta
from metriclearning.datasets.data_sample import DataSample, Metriclearn_array


class MKernel(metaclass=ABCMeta):
    """
    Abstract class MKL and MVML should inherit from
    for methods of transform kernel to/from data
    """

    def _get_kernel(self, X, Y=None, v=0):
        if self.kernel_params is not None:
            if isinstance(self.kernel_params, list):
                ind = min(v, len(self.kernel) - 1)
                params = self.kernel_params[v]
            else:
                params = self.kernel_params
        else:
            params = {}
        if isinstance(self.kernel, str):
            met = self.kernel
        elif isinstance(self.kernel, list):
            ind = min(v, len(self.kernel) - 1)
            met = self.kernel[ind]
        # Y,
        return pairwise_kernels(X, Y, metric=met,
                                filter_params=True, **params)

    def _global_kernel_transform(self, X, views_ind=None, Y=None):
        kernel_dict = {}

        X_ = None
        if Y is None:
            y = Y
        if isinstance(X, np.ndarray) and X.ndim == 1:
            X_= Metriclearn_array(X, views_ind)
            for v in range(X.shape[0]):
                if Y is not None:  y = Y.get_view(v) #  y = self._global_check_pairwise(X_, Y, v)
                kernel_dict[v] = self._get_kernel(X[v], y)
        elif isinstance(X, dict):
            X_= Metriclearn_array(X)
            for v in X.keys():
                if Y is not None:  y = Y.get_view(v) # y = self._global_check_pairwise(X_, Y, v)
                kernel_dict[v] = self._get_kernel(X[v], y)
        elif isinstance(X, np.ndarray) and X.ndim > 1:
            X_ = Metriclearn_array(X, views_ind)
            X = X_
        if isinstance(X, Metriclearn_array):
            for v in range(X.n_views):
                if Y is not None:   y = Y.get_view(v) # y = self._global_check_pairwise(X, Y, v)
                kernel_dict[v] = self._get_kernel(X.get_view(v), y)
            X_= X
        if not isinstance(X_, Metriclearn_array):
            raise TypeError("Input format is not reconized")
        K_ = Metriclearn_array(kernel_dict)
        return X_, K_

    def _calc_nystrom(self, kernels, n_approx):
        # calculates the nyström approximation for all the kernels in the given dictionary
        self.W_sqrootinv_dict = {}
        self.U_dict = {}
        for v in range(kernels.n_views):
            kernel = kernels.get_view(v)
            E = kernel[:, 0:n_approx]
            W = E[0:n_approx, :]
            Ue, Va, _ = sp.linalg.svd(W)
            vak = Va[0:n_approx]
            inVa = np.diag(vak ** (-0.5))
            U_v = np.dot(E, np.dot(Ue[:, 0:n_approx], inVa))
            self.U_dict[v] = U_v
            self.W_sqrootinv_dict[v] = np.dot(Ue[:, 0:n_approx], inVa)