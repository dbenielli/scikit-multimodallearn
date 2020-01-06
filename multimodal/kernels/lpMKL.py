import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y
from sklearn.utils.validation  import check_array
from sklearn.utils.validation  import check_is_fitted
from metriclearning.mkernel import MKernel


class MKL(BaseEstimator, ClassifierMixin, MKernel):
    """
    MKL Classifier for multiview learning

    Parameters
    ----------

    lmbda : float coeficient for combined kernels

    m_param : float (default : 1.0)
       value between 0 and 1 indicating level of nyström approximation;
       1 = no approximation

    kernel : list of str (default: "precomputed") if kernel is as input of fit function set kernel to
             "precomputed"
             list or str indicate the metrics used for each kernels
             list of pairwise kernel function name
             (default : "precomputed") if kernel is as input of fit function set kernel to "precomputed"
             example : ['rbf', 'additive_chi2', 'linear' ] for function defined in as
             PAIRWISE_KERNEL_FUNCTIONS

    kernel_params : list of str default : None) list of dictionaries for parameters of kernel [{'gamma':50}
                    list of dict of corresponding kernels params KERNEL_PARAMS

    use_approx : (default : True) to use approximation of m_param < 1

    n_loops : (default 50) number of iterions

    Attributes
    ----------
    lmbda : float coeficient for combined kernels

    m_param : float (default : 1.0)
       value between 0 and 1 indicating level of nyström approximation;
       1 = no approximation

    kernel : list or str indicate the metrics used for each kernels
             list of pairwise kernel function name
             (default : "precomputed")
             example : ['rbf', 'additive_chi2', 'linear' ] for function defined in as
             PAIRWISE_KERNEL_FUNCTIONS
             example kernel=['rbf', 'rbf'], for the first two views

    kernel_params: list of dict of corresponding kernels params KERNEL_PARAMS

    precision : float (default : 1E-4) precision to stop algorithm

    n_loops : number of iterions

    classes_ : array like unique label for classes

    X_ : :class:`metriclearning.datasets.data_sample.Metriclearn_array` array of input sample

    K_ : :class:`metriclearning.datasets.data_sample.Metriclearn_array` array of processed kernels

    y_ : array-like, shape = (n_samples,)
         Target values (class labels).

    C : learning solution that is learned in MKL

    weights : learned weight for combining the solutions of views, learned in

    """
    def __init__(self, lmbda, m_param=1.0, kernel="precomputed",
                 kernel_params=None, use_approx=True, precision=1E-4, n_loops=50):
        # calculate nyström approximation (if used)
        self.lmbda = lmbda
        self.n_loops = n_loops
        self.use_approx = use_approx
        self.m_param = m_param
        self.kernel= kernel
        self.kernel_params = kernel_params
        self.precision = precision

    def fit(self, X, y= None, views_ind=None):
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
        self.X_, self.K_ = self._global_kernel_transform(X, views_ind=views_ind)
        self.classes_ = unique_labels(y)
        check_X_y(self.X_, y)
        self.y_ = y
        n = self.K_.shape[0]
        self._calc_nystrom(self.K_, n)
        C, weights = self.learn_lpMKL()
        self.C = C
        self.weights = weights
        return self

    def learn_lpMKL(self):
        """
        function of lpMKL learning

        Returns
        -------
        return tuple (C, weights)
        """
        views = self.K_.n_views
        X = self.K_
        p = 2
        n = self.K_.shape[0]
        weights = np.ones(views) / (views)

        prevalpha = False
        max_diff = 1
        if (self.precision >= max_diff):
            raise ValueError(" %f precision must be > to %f" % (self.precision,max_diff))
        kernels = np.zeros((views, n, n))
        for v in range(0, views):
            kernels[v, :, :] = np.dot(self.U_dict[v], np.transpose(self.U_dict[v]))

        rounds = 0
        stuck = False
        while max_diff > self.precision and rounds < self.n_loops and not stuck:

            # gammas are fixed upon arrival to the loop
            # -> solve for alpha!

            if self.m_param < 1 and self.use_approx:
                combined_kernel = np.zeros((n, n))
                for v in range(0, views):
                    combined_kernel = combined_kernel + weights[v] * kernels[v]
            else:
                combined_kernel = np.zeros((n, n))
                for v in range(0, views):
                    combined_kernel = combined_kernel + weights[v]*X.get_view(v)
            # combined kernel includes the weights

            # alpha = (K-lambda*I)^-1 y
            C = np.linalg.solve((combined_kernel + self.lmbda * np.eye(n)), self.y_)

            # alpha fixed -> calculate gammas
            weights_old = weights.copy()

            # first the ||f_t||^2 todo wtf is the formula used here????
            ft2 = np.zeros(views)
            for v in range(0, views):
                if self.m_param < 1 and self.use_approx:
                        # ft2[v,vv] = weights_old[v,vv] * np.dot(np.transpose(C), np.dot(np.dot(np.dot(data.U_dict[v],
                        #                                                             np.transpose(data.U_dict[v])),
                        #                                                             np.dot(data.U_dict[vv],
                        #                                                             np.transpose(data.U_dict[vv]))), C))
                    ft2[v] = np.linalg.norm(weights_old[v] * np.dot(kernels[v], C))**2
                else:
                    ft2[v] = np.linalg.norm(weights_old[v] * np.dot(X.get_view(v), C))**2
                    # ft2[v] = weights_old[v] * np.dot(np.transpose(C), np.dot(data.kernel_dict[v], C))
            # calculate the sum for downstairs
            # print(weights_old)
            # print(ft2)
            # print(ft2 ** (p / (p + 1.0)))
            downstairs = np.sum(ft2 ** (p / (p + 1.0))) ** (1.0 / p)
            # and then the gammas
            weights = (ft2 ** (1 / (p + 1))) / downstairs

            # convergence
            if prevalpha == False:  # first time in loop we don't have a previous alpha value
                prevalpha = True
                diff_alpha = 1
            else:
                diff_alpha = np.linalg.norm(C_old - C) / np.linalg.norm(C_old)
                max_diff_gamma_prev = max_diff_gamma

            max_diff_gamma = np.max(np.max(np.abs(weights - weights_old)))

            # Add to prevent faillure on max_diff_gamma_prev
            if not 'max_diff_gamma_prev' in globals(): max_diff_gamma_prev = max_diff_gamma
            # try to see if convergence is as good as it gets: if it is stuck
            if max_diff_gamma < 10*self.precision and max_diff_gamma_prev < max_diff_gamma:
                # if the gamma difference starts to grow we are most definitely stuck!
                # (this condition determined empirically by running algo and observing the convergence)
                stuck = True
            if rounds > 1 and max_diff_gamma - max_diff_gamma_prev > 100*self.precision:
                # If suddenly the difference starts to grow much
                stuck = True

            max_diff = np.max([max_diff_gamma, diff_alpha])
            # print([max_diff_gamma, diff_alpha])  # print if convergence is interesting
            C_old = C.copy()
            rounds = rounds + 1
        # print("\nlearned the weights:")
        # np.set_printoptions(precision=3, suppress=True)
        # print(weights)
        # print("")

        # print if resulting convergence is of interest
        # print("convergence of ", max_diff, " at step ", rounds, "/500")
        if stuck:
            return C_old, weights_old
        else:
            return C, weights


    def predict(self, X, views_ind=None):
        """

        Parameters
        ----------

        X : - Metriclearn_array {array-like, sparse matrix}, shape = (n_samples, n_features)
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
        check_is_fitted(self, ['X_', 'C', 'K_', 'y_', 'weights'])
        X , test_kernels = self._global_kernel_transform(X,
                                                         views_ind=views_ind,
                                                         Y=self.X_)
        check_array(X)
        C = self.C
        weights  = self.weights
        return self.lpMKL_predict(test_kernels, C, weights)


    def lpMKL_predict(self, X, C, weights):
        """

        Parameters
        ----------

        X : array-like test kernels precomputed array like

        C : corresponding to  Confusion learned matrix

        weights : learned weights

        Returns
        -------

        y : numpy.ndarray, shape = (n_samples,)
            Predicted classes.
        """
        views = X.n_views
        tt = X.shape[0]
        m = self.K_.shape[0] # self.m_param * n

        #  NO TEST KERNEL APPROXIMATION
        # kernel = weights[0] * self.data.test_kernel_dict[0]
        # for v in range(1, views):
        #     kernel = kernel + weights[v] * self.data.test_kernel_dict[v]

        # TEST KERNEL APPROXIMATION
        kernel = np.zeros((tt, self.K_.shape[0]))
        for v in range(0, views):
            if self.m_param < 1:
                kernel = kernel + weights[v] * np.dot(np.dot(X.get_view(v)[:, 0:m], self.W_sqrootinv_dict[v]),
                                                  np.transpose(self.U_dict[v]))
            else:
                kernel = kernel + weights[v] * X.get_view(v)

        return np.dot(kernel, C)



