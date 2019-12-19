import numpy as np
import scipy.linalg as spli
from scipy.sparse.linalg import splu
from scipy.sparse import csc_matrix
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils.validation import check_X_y
from sklearn.utils.validation  import check_array
from sklearn.metrics.pairwise import check_pairwise_arrays
from sklearn.utils.validation  import check_is_fitted
from metriclearning.datasets.data_sample import DataSample, Metriclearn_array
from metriclearning.mkernel import MKernel

"""
    Copyright (C) 2018  Riikka Huusari

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.


This file contains algorithms for Multi-View Metric Learning (MVML) as introduced in

Riikka Huusari, Hachem Kadri and Cécile Capponi:
Multi-View Metric Learning in Vector-Valued Kernel Spaces
in International Conference on Artificial Intelligence and Statistics (AISTATS) 2018

Usage (see also demo.py for a more detailed example):
    create a MVML object via:
        mvml = MVML(kernel_dict, label_vector, regression_parameter_list, nystrom_param)
    learn the model:
        A, g, w = mvml.learn_mvml()
    predict with the model:
        predictions = predict_mvml(test_kernel_dict, g, w)

(parameter names as in the paper)

Code is tested with Python 3.5.2 and numpy 1.12.1
"""


class MVML(MKernel, BaseEstimator, ClassifierMixin):
    r"""
    The MVML Classifier

    Parameters
    ----------
    regression_params: array/list of regression parameters, first for basic regularization, second for
        regularization of A (not necessary if A is not learned)

    kernel : list of str (default: "precomputed") if kernel is as input of fit function set kernel to
             "precomputed"
             list or str indicate the metrics used for each kernels
             list of pairwise kernel function name
             (default : "precomputed") if kernel is as input of fit function set kernel to "precomputed"
             example : ['rbf', 'additive_chi2', 'linear' ] for function defined in as
             PAIRWISE_KERNEL_FUNCTIONS

    kernel_params : list of str default : None) list of dictionaries for parameters of kernel [{'gamma':50}
                    list of dict of corresponding kernels params KERNEL_PARAMS

    nystrom_param: value between 0 and 1 indicating level of nyström approximation; 1 = no approximation

    learn_A :  integer (default 1) choose if A is learned or not: 1 - yes (default);
               2 - yes, sparse; 3 - no (MVML_Cov); 4 - no (MVML_I)

    learn_w : integer (default 0) where learn w is needed

    precision : float (default : 1E-4) precision to stop algorithm

    n_loops : (default 6) number of iterions


    Attributes
    ----------
    regression_params : array/list of regression parameters

    kernel : list or str indicate the metrics used for each kernels
             list of pairwise kernel function name
             (default : "precomputed")
             example : ['rbf', 'additive_chi2', 'linear' ] for function defined in as
             PAIRWISE_KERNEL_FUNCTIONS
             example kernel=['rbf', 'rbf'], for the first two views

    kernel_params: list of dict of corresponding kernels params KERNEL_PARAMS

    learn_A :  1 where Learn matrix A is needded

    learn_w : integer where learn w is needed

    precision : float (default : 1E-4) precision to stop algorithm

    n_loops : number of itterions

    n_approx : number of samples in approximation, equals n if no approx.

    classes_ : array like unique label for classes

    warning_message : dictionary with warning messages

    X_ : :class:`metriclearning.datasets.data_sample.Metriclearn_array` array of input sample

    K_ : :class:`metriclearning.datasets.data_sample.Metriclearn_array` array of processed kernels

    y_ : array-like, shape = (n_samples,)
         Target values (class labels).

    """
    # r_cond = 10-30
    def __init__(self, lmbda, eta, nystrom_param, kernel="precomputed",
                 kernel_params=None,
                 learn_A=1, learn_w=0, precision=1E-4, n_loops=6):

        # calculate nyström approximation (if used)
        self.nystrom_param = nystrom_param
        self.lmbda = lmbda
        self.eta = eta
        # self.regression_params = regression_params
        self.learn_A = learn_A
        self.learn_w = learn_w
        self.n_loops = n_loops
        self.kernel= kernel
        self.kernel_params = kernel_params
        self.precision = precision
        self.warning_message = {}

    def fit(self, X, y= None, views_ind=None):
        """
        Fit the MVML classifier

        Parameters
        ----------

        X : - Metriclearn_array {array-like, sparse matrix}, shape = (n_samples, n_features)
              Training multi-view input samples. can be also Kernel where attibute 'kernel'
              is set to precompute "precomputed"
            or
            - Dictionary of {array like} with shape = (n_samples, n_features)  for multi-view
              for each view.
            - Array of {array like} with shape = (n_samples, n_features)  for multi-view
              for each view.
            - {array like} with (n_samples, nviews *  n_features) with 'views_ind' diferent to 'None'


        y : array-like, shape = (n_samples,)
            Target values (class labels).
            array of length n_samples containing the classification/regression labels
            for training data

        views_ind : array-like (default=[0, n_features//2, n_features])
            Paramater specifying how to extract the data views from X:

            - views_ind is a 1-D array of sorted integers, the entries
              indicate the limits of the slices used to extract the views,
              where view ``n`` is given by
              ``X[:, views_ind[n]:views_ind[n+1]]``.

              With this convention each view is therefore a view (in the NumPy
              sense) of X and no copy of the data is done.


        Returns
        -------
        self : object
            Returns self.
        """
        # Check that X and y have correct shape

        # Store the classes seen during fit

        self.X_, self.K_= self._global_kernel_transform(X, views_ind=views_ind)
        check_X_y(self.X_, y)

        self.classes_ = unique_labels(y)
        self.y_ = y

        # n = X[0].shape[0]
        n = self.K_.shape[0]
        self.n_approx = int(np.floor(self.nystrom_param * n))  # number of samples in approximation, equals n if no approx.
        if self.nystrom_param < 1:
            self._calc_nystrom(self.K_, self.n_approx)
        else:
            self.U_dict = self.K_._todict()

        # Return the classifier
        self.learn_mvml(learn_A=self.learn_A, learn_w=self.learn_w, n_loops=self.n_loops)
        if self.warning_message:
            print("warning appears during fit process", self.warning_message)
        return self

    def learn_mvml(self, learn_A=1, learn_w=0, n_loops=6):
        """

        Parameters
        ----------
        learn_A: int choose if A is learned or not (default: 1):
                 1 - yes (default);
                 2 - yes, sparse;
                 3 - no (MVML_Cov);
                 4 - no (MVML_I)
        learn_w: int choose if w is learned or not (default: 0):
                 0 - no (uniform 1/views, default setting),
                 1 - yes
        n_loops: int maximum number of iterations in MVML, (default: 6)
                 usually something like default 6 is already converged

        Returns
        -------
        tuple (A, g, w) with A (metrcic matrix - either fixed or learned),
                             g (solution to learning problem),
                             w (weights - fixed or learned)
        """
        views = len(self.U_dict)
        n = self.U_dict[0].shape[0]
        lmbda = self.lmbda
        if learn_A < 3:
            eta = self.eta

        # ========= initialize A =========

        # positive definite initialization (with multiplication with the U matrices if using approximation)
        A = np.zeros((views * self.n_approx, views * self.n_approx))
        if learn_A < 3:
            for v in range(views):
                if self.nystrom_param < 1:
                    A[v * self.n_approx:(v + 1) * self.n_approx, v * self.n_approx:(v + 1) * self.n_approx] = \
                        np.dot(np.transpose(self.U_dict[v]), self.U_dict[v])
                else:
                    A[v * self.n_approx:(v + 1) * self.n_approx, v * self.n_approx:(v + 1) * self.n_approx] = np.eye(n)
        # otherwise initialize like this if using MVML_Cov
        elif learn_A == 3:
            for v in range(views):
                for vv in range(views):
                    if self.nystrom_param < 1:
                        A[v * self.n_approx:(v + 1) * self.n_approx, vv * self.n_approx:(vv + 1) * self.n_approx] = \
                            np.dot(np.transpose(self.U_dict[v]), self.U_dict[vv])
                    else:
                        A[v * self.n_approx:(v + 1) * self.n_approx, vv * self.n_approx:(vv + 1) * self.n_approx] = \
                            np.eye(n)
        # or like this if using MVML_I
        elif learn_A == 4:
            for v in range(views):
                if self.nystrom_param < 1:
                    A[v * self.n_approx:(v + 1) * self.n_approx, v * self.n_approx:(v + 1) * self.n_approx] = \
                        np.eye(self.n_approx)
                else:
                    # it might be wise to make a dedicated function for MVML_I if using no approximation
                    # - numerical errors are more probable this way using inverse
                    A[v * self.n_approx:(v + 1) * self.n_approx, v * self.n_approx:(v + 1) * self.n_approx] = \
                        spli.pinv(self.U_dict[v])  # U_dict holds whole kernels if no approx

        # ========= initialize w, allocate g =========
        w = (1 / views) * np.ones((views, 1))
        g = np.zeros((views * self.n_approx, 1))

        # ========= learn =========
        loop_counter = 0
        while True:
            if loop_counter > 0:
                g_prev = np.copy(g)
                A_prev = np.copy(A)
                w_prev = np.copy(w)

            # ========= update g =========

            # first invert A
            try:
                # A_inv = np.linalg.pinv(A + 1e-09 * np.eye(views * self.n_approx))
                cond_A = np.linalg.cond(A + 1e-08 * np.eye(views * self.n_approx))
                if cond_A < 10:
                    A_inv = spli.pinv(A + 1e-8 * np.eye(views * self.n_approx))
                else:
                    # A_inv = self._inverse_precond_LU(A + 1e-8 * np.eye(views * self.n_approx), pos="precond_A") # self._inverse_precond_jacobi(A + 1e-8 * np.eye(views * self.n_approx), pos="precond_A")
                    A_inv = self._inv_best_precond(A + 1e-8 * np.eye(views * self.n_approx), pos="precond_A")
            except spli.LinAlgError:
                self.warning_message["LinAlgError"] = self.warning_message.get("LinAlgError", 0) + 1
                try:
                    A_inv = spli.pinv(A + 1e-07 * np.eye(views * self.n_approx))
                except spli.LinAlgError:
                    try:
                        A_inv = spli.pinv(A + 1e-06 * np.eye(views * self.n_approx)) # , rcond=self.r_cond*minA
                    except ValueError:
                        self.warning_message["ValueError"] = self.warning_message.get("ValueError", 0) + 1
                        return A_prev, g_prev
            except ValueError:
                self.warning_message["ValueError"] = self.warning_message.get("ValueError", 0) + 1
                return A_prev, g_prev
            # print("A_inv ",np.sum(A_inv))
            # then calculate g (block-sparse multiplications in loop) using A_inv
            for v in range(views):
                for vv in range(views):
                    A_inv[v * self.n_approx:(v + 1) * self.n_approx, vv * self.n_approx:(vv + 1) * self.n_approx] = \
                        w[v] * w[vv] * np.dot(np.transpose(self.U_dict[v]), self.U_dict[vv]) + \
                        lmbda * A_inv[v * self.n_approx:(v + 1) * self.n_approx,
                                      vv * self.n_approx:(vv + 1) * self.n_approx]
                g[v * self.n_approx:(v + 1) * self.n_approx, 0] = np.dot(w[v] * np.transpose(self.U_dict[v]), self.y_)
            try:
                # minA_inv = np.min(np.absolute(A_inv)) , rcond=self.r_cond*minA_inv
                # here A_inv isn't actually inverse of A (changed in above loop)
                if np.linalg.cond(A_inv) < 10:
                   g = np.dot(spli.pinv(A_inv), g)
                else:
                    # g = np.dot(self._inverse_precond_LU(A_inv, pos="precond_A_1"), g)
                    g = np.dot(self._inv_best_precond(A_inv, pos="precond_A_1"), g)
            except spli.LinAlgError:
                self.warning_message["LinAlgError"] = self.warning_message.get("LinAlgError", 0) + 1
                g = spli.solve(A_inv, g)

            # ========= check convergence =========

            if learn_A > 2 and learn_w != 1:  # stop at once if only g is to be learned
                break

            if loop_counter > 0:

                # convergence criteria
                g_diff = np.linalg.norm(g - g_prev) / np.linalg.norm(g_prev)
                A_diff = np.linalg.norm(A - A_prev, ord='fro') / np.linalg.norm(A_prev, ord='fro')
                if g_diff < self.precision and A_diff < self.precision:
                    break

            if loop_counter >= n_loops:  # failsafe
                break

            # ========= update A =========
            if learn_A == 1:
                A = self._learn_A_func(A, g, lmbda, eta)
            elif learn_A == 2:
                A = self._learn_blocksparse_A(A, g, views, self.n_approx, lmbda, eta)

            # ========= update w =========
            if learn_w == 1:
                Z = np.zeros((n, views))
                for v in range(views):
                    Z[:, v] = np.dot(self.U_dict[v], g[v * self.n_approx:(v + 1) * self.n_approx]).ravel()
                w = np.dot(spli.pinv(np.dot(np.transpose(Z), Z)), np.dot(np.transpose(Z), self.y_))
            loop_counter += 1
        self.g = g
        self.w = w
        self.A = A
        return A, g, w

    def _inv_best_precond(self, A, pos="precond_A"):
        J_1 = np.diag(1.0/np.diag(A))
        Pre_J = np.dot(J_1, A)
        Pm, L, U = spli.lu(A)
        M = spli.inv(np.dot(L, U))
        Pre_lu = np.dot(M, A)
        # print("cond a", np.linalg.cond(A))
        # print("cond Pre_J", np.linalg.cond(Pre_J))
        # print("cond Pre_lu", np.linalg.cond(Pre_lu))
        if np.linalg.cond(A) > np.linalg.cond(Pre_J) and np.linalg.cond(Pre_J) <= np.linalg.cond(Pre_lu):
            P_inv = spli.pinv(Pre_J)
            A_inv = np.dot(P_inv,  J_1)
            self.warning_message[pos] = self.warning_message.get(pos, 0) + 1
        elif  np.linalg.cond(Pre_lu) < np.linalg.cond(A):
            P_inv = spli.pinv(Pre_lu)
            A_inv = np.dot(P_inv,  M)
            self.warning_message[pos] = self.warning_message.get(pos, 0) + 1
        else:
            A_inv = spli.pinv(A)
        return A_inv

    def _inverse_precond_jacobi(self, A, pos="precond_A"):
        J_1 = np.diag(1.0/np.diag(A))
        # J_1 = np.linalg.inv(J)
        P = np.dot(J_1, A)
        if np.linalg.cond(A) > np.linalg.cond(P):
            P_inv = spli.pinv(P)
            A_inv = np.dot(P_inv,  J_1)
            self.warning_message[pos] = self.warning_message.get(pos, 0) + 1
        else:
            A_inv = self._inverse_precond_LU(A, pos=pos)

        return A_inv

    def _inverse_precond_LU(self, A, pos="precond_A"):
        P, L, U = spli.lu(A)
        M = spli.inv(np.dot(L, U))
        P = np.dot(M, A)
        if np.linalg.cond(A) > np.linalg.cond(P):
            P_inv = spli.pinv(P)
            A_inv = np.dot(P_inv,  M)
            self.warning_message[pos] = self.warning_message.get(pos, 0) + 1
        else:
            A_inv = spli.pinv(A)
        return A_inv

    def predict(self, X, views_ind=None):
        """

        Parameters
        ----------
        X : different formats are supported
            - Metriclearn_array {array-like, sparse matrix}, shape = (n_samples, n_features)
              Training multi-view input samples. can be also Kernel where attibute 'kernel'
              is set to precompute "precomputed"

            - Dictionary of {array like} with shape = (n_samples, n_features)  for multi-view
              for each view.
            - Array of {array like} with shape = (n_samples, n_features)  for multi-view
              for each view.
            - {array like} with (n_samples, nviews *  n_features) with 'views_ind' diferent to 'None'

        views_ind : array-like (default=[0, n_features//2, n_features])
            Paramater specifying how to extract the data views from X:

            - views_ind is a 1-D array of sorted integers, the entries
              indicate the limits of the slices used to extract the views,
              where view ``n`` is given by
              ``X[:, views_ind[n]:views_ind[n+1]]``.

              With this convention each view is therefore a view (in the NumPy
              sense) of X and no copy of the data is done.

        Returns
        -------
        y : numpy.ndarray, shape = (n_samples,)
            Predicted classes.
        """
        check_is_fitted(self, ['X_', 'U_dict', 'K_', 'y_']) # , 'U_dict', 'K_' 'y_'
        X , test_kernels = self._global_kernel_transform(X,
                                                         views_ind=views_ind,
                                                         Y=self.X_)
        check_array(X)
        return self.predict_mvml(test_kernels, self.g, self.w)

    def predict_mvml(self, test_kernels, g, w):
        """

        Parameters
        ----------
        test_kernels : `Metriclearn_array` of test kernels

        g : learning solution that is learned in learn_mvml

        w : weights for combining the solutions of views, learned in learn_mvml

        Returns
        -------
        numpy.ndarray, shape = (n_samples,) of test_kernels
            Predicted classes.

        """
        views = len(self.U_dict)
        # t = test_kernels[0].shape[0]
        t = test_kernels.shape[0]
        K = np.zeros((t, views * self.n_approx))
        for v in range(views):
            if self.nystrom_param < 1:
                K[:, v * self.n_approx:(v + 1) * self.n_approx] = w[v] * \
                                                                  np.dot(test_kernels.get_view(v)[:, 0:self.n_approx],
                                                                         self.W_sqrootinv_dict[v])
            else:
                K[:, v * self.n_approx : (v + 1) * self.n_approx] = w[v] * test_kernels.get_view(v)

        return np.dot(K, g)

    def _learn_A_func(self, A, g, lmbda, eta):
        # basic gradient descent
        stepsize = 0.5
        if stepsize*eta >= 0.5:
            stepsize = 0.9*(1/(2*eta))  # make stepsize*eta < 0.5

        loops = 0
        not_converged = True
        while not_converged:
            A_prev = np.copy(A)
            # minA = np.min(np.absolute(A)) , rcond=self.r_cond*minA
            A_pinv = spli.pinv(A)
            A = (1-2*stepsize*eta)*A + stepsize*lmbda*np.dot(np.dot(A_pinv, g), np.dot(np.transpose(g), A_pinv))

            if loops > 0:
                prev_diff = diff
            diff = np.linalg.norm(A - A_prev) / np.linalg.norm(A_prev)
            if loops > 0 and prev_diff > diff:
                A = A_prev
                stepsize = stepsize*0.1
            if diff < 1e-5:
                not_converged = False
            if loops > 100:
                not_converged = False
            loops += 1

        return A

    def _learn_blocksparse_A(self, A, g, views, m, lmbda, eta):

        # proximal gradient update method
        converged = False
        rounds = 0

        L = lmbda * np.linalg.norm(np.dot(g, g.T))
        # print("L ", L)

        while not converged and rounds < 100:
            # no line search - this has worked well enough experimentally
            A = self._proximal_update(A, views, m, L, g, lmbda, eta)

            # convergence
            if rounds > 0:
                A_diff = np.linalg.norm(A - A_prev) / np.linalg.norm(A_prev)

                if A_diff < 1e-3:
                    converged = True
            A_prev = np.copy(A)
            rounds += 1

        return A

    def _proximal_update(self, A_prev, views, m, L, D, lmbda, gamma):

        # proximal update

        # the inverse is not always good to compute - in that case just return the previous one and end the search
        try:
            # minA_inv = np.min(np.absolute(A_prev)) , rcond=self.r_cond*minA_inv
            A_prev_inv = spli.pinv(A_prev)
        except spli.LinAlgError:
            try:
                A_prev_inv = spli.pinv(A_prev + 1e-6 * np.eye(views * m))
            except spli.LinAlgError:
                return A_prev
            except ValueError:
                return A_prev
        except ValueError:
            return A_prev

        if np.any(np.isnan(A_prev_inv)):
            # just in case the inverse didn't return a proper solution (happened once or twice)
            return A_prev

        A_tmp = A_prev + (lmbda / L) * np.dot(np.dot(A_prev_inv.T, D), np.dot(np.transpose(D), A_prev_inv.T))

        # if there is one small negative eigenvalue this gets rid of it
        try:
            val, vec = spli.eigh(A_tmp)
        except spli.LinAlgError:
            return A_prev
        except ValueError:
            return A_prev
        val[val < 0] = 0

        A_tmp = np.dot(vec, np.dot(np.diag(val), np.transpose(vec)))
        A_new = np.zeros((views*m, views*m))

        # proximal update, group by group (symmetric!)
        for v in range(views):
            for vv in range(v + 1):
                if v != vv:
                    if np.linalg.norm(A_tmp[v * m:(v + 1) * m, vv * m:(vv + 1) * m]) != 0:
                        multiplier = 1 - gamma / (2 * np.linalg.norm(A_tmp[v * m:(v + 1) * m, vv * m:(vv + 1) * m]))
                        if multiplier > 0:
                            A_new[v * m:(v + 1) * m, vv * m:(vv + 1) * m] = multiplier * A_tmp[v * m:(v + 1) * m,
                                                                                               vv * m:(vv + 1) * m]
                            A_new[vv * m:(vv + 1) * m, v * m:(v + 1) * m] = multiplier * A_tmp[vv * m:(vv + 1) * m,
                                                                                               v * m:(v + 1) * m]
                else:
                    if (np.linalg.norm(A_tmp[v * m:(v + 1) * m, v * m:(v + 1) * m])) != 0:
                        multiplier = 1 - gamma / (np.linalg.norm(A_tmp[v * m:(v + 1) * m, v * m:(v + 1) * m]))
                        if multiplier > 0:
                            A_new[v * m:(v + 1) * m, v * m:(v + 1) * m] = multiplier * A_tmp[v * m:(v + 1) * m,
                                                                                             v * m:(v + 1) * m]

        return A_new
