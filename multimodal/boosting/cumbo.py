# -*- coding: utf-8 -*-
r"""

This module contains a **Mu**\ lti\ **C**\ onfusion **M**\ Matrix **B**\ osting (**CuMBo**)
estimator for classification implemented in the ``MuCumboClassifier`` class.
"""

# Université d'Aix Marseille (AMU) -
# Centre National de la Recherche Scientifique (CNRS) -
# Université de Toulon (UTLN).
# Copyright © 2017-2018 AMU, CNRS, UTLN
#
# This file is part of multimodalboost.
#
# multimodalboost is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# multimodalboost is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with multiconfusion. If not, see <http://www.gnu.org/licenses/>.
#
# Author: Dominique Benielli- Laboratoire d'Informatique et Systèmes - UMR 7020

# The implementation of the MuCumboClassifier in this module used the code of
# sklearn.ensemble.AdaBoostClassifier as a model and tried to use the same
# structure, notations and behavior where possible.

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.ensemble import BaseEnsemble
from sklearn.ensemble.forest import BaseForest
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree._tree import DTYPE
from sklearn.tree.tree import BaseDecisionTree
from sklearn.utils import check_array, check_X_y, check_random_state
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, has_fit_parameter
from cvxopt import solvers, matrix, spdiag, exp, spmatrix, mul, div
from .boost import UBoosting
import warnings


class MuCumboClassifier(BaseEnsemble, ClassifierMixin, UBoosting):
    r"""It then iterates the process on the same dataset but where the weights of
    incorrectly classified instances are adjusted such that subsequent
    classifiers focus more on difficult cases.
    A MuCoMBo classifier.

    A MuMBo classifier is a meta-estimator that implements a multimodal
    (or multi-view) boosting algorithm:

    It fits a set of classifiers on the original dataset splitted into several
    views and retains the classifier obtained for the best view.

    This class implements the MuMBo algorithm [1]_.

    Parameters
    ----------
    base_estimator : object, optional (default=DecisionTreeClassifier)
        Base estimator from which the boosted ensemble is built.
        Support for sample weighting is required, as well as proper `classes_`
        and `n_classes_` attributes. The default is a DecisionTreeClassifier
        with parameter ``max_depth=1``.

    n_estimators : integer, optional (default=50)
        Maximum number of estimators at which boosting is terminated.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.


    Attributes
    ----------
    estimators\_ : list of classifiers
        Collection of fitted sub-estimators.

    classes\_ : numpy.ndarray, shape = (n_classes,)
        Classes labels.

    n_classes\_ : int
        Number of classes.

    n_views\_ : int
        Number of views

    estimator_weights\_ : numpy.ndarray of floats, shape = (len(estimators\_),)
        Weights for each estimator in the boosted ensemble.

    estimator_errors_ : array of floats
        Empirical loss for each iteration.


    best_views\_ : numpy.ndarray of integers, shape = (len(estimators\_),)
        Indices of the best view for each estimator in the boosted ensemble.

    n_yi : numpy ndarray of int contains number of train sample for each classe shape (n_classes,)

    Examples
    --------
    >>> from multimodal.boosting.cumbo import MuCumboClassifier
    >>> from sklearn.datasets import load_iris
    >>> X, y = load_iris(return_X_y=True)
    >>> views_ind = [0, 2, 4]  # view 0: sepal data, view 1: petal data
    >>> clf = MuCumboClassifier(random_state=0)
    >>> clf.fit(X, y, views_ind)  # doctest: +NORMALIZE_WHITESPACE

    >>> print(clf.predict([[ 5.,  3.,  1.,  1.]]))
    [1]
    >>> views_ind = [[0, 2], [1, 3]]  # view 0: length data, view 1: width data
    >>> clf = MuCumboClassifier(random_state=0)
    >>> clf.fit(X, y, views_ind)  # doctest: +NORMALIZE_WHITESPACE

    >>> print(clf.predict([[ 5.,  3.,  1.,  1.]]))
    [1]

    >>> from sklearn.tree import DecisionTreeClassifier
    >>> base_estimator = DecisionTreeClassifier(max_depth=2)
    >>> clf = MuCumboClassifier(base_estimator=base_estimator, random_state=0)
    >>> clf.fit(X, y, views_ind)  # doctest: +NORMALIZE_WHITESPACE

    >>> print(clf.predict([[ 5.,  3.,  1.,  1.]]))
    [1]

    See also
    --------
    sklearn.ensemble.AdaBoostClassifier,
    sklearn.ensemble.GradientBoostingClassifier,
    sklearn.tree.DecisionTreeClassifier

    References
    ----------

    .. [1] Ko\c{c}o, Sokol and Capponi, C{\'e}cile
           A Boosting Approach to Multiview Classification with Cooperation,
           2011,Proceedings of the 2011 European Conference on Machine Learning
           and Knowledge Discovery in Databases - Volume Part II, 209--228 Springer-Verlag
           https://link.springer.com/chapter/10.1007/978-3-642-23783-6_1

    .. [2] Sokol Koço,
           "Tackling the uneven views problem with cooperation based ensemble
           learning methods",
           PhD Thesis, Aix-Marseille Université, 2013,
           http://www.theses.fr/en/2013AIXM4101.
    """

    def __init__(self,
                 base_estimator=None,
                 n_estimators=50,
                 random_state=None): # n_estimators=50,
        super(MuCumboClassifier, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators)
        self.random_state = random_state

    def _validate_estimator(self):
        """Check the estimator and set the base_estimator_ attribute."""
        super(MuCumboClassifier, self)._validate_estimator(
            default=DecisionTreeClassifier(max_depth=1))

        if not has_fit_parameter(self.base_estimator_, "sample_weight"):
            raise ValueError("%s doesn't support sample_weight."
                             % self.base_estimator_.__class__.__name__)

    def _init_var(self, n_views, y):
        "Create and initialize the variables used by the MuMBo algorithm."
        n_classes = self.n_classes_
        n_samples = y.shape[0]
        # n_yi = np.unique(y, return_inverse=True)
        cost = np.ones((n_views, n_samples, n_classes))
        score_function = np.zeros((n_views, n_samples, n_classes))
        n_yi_s = np.zeros(n_classes, dtype=np.int)
        for indice_class in range(n_classes):
            # n_yi number of examples of the class y_i
            n_yi = np.where(y==indice_class)[0].shape[0]
            n_yi_s[indice_class] = int(n_yi)
            cost[:, :, indice_class] /=   n_yi
        cost[:, np.arange(n_samples), y] *= -(n_classes-1)
        label_score = np.zeros((n_views, n_samples, n_classes))
        label_score_global = np.zeros((n_samples, n_classes))
        predicted_classes = np.empty((n_views, n_samples), dtype=np.int64)
        beta_class = np.ones((n_views, n_classes)) / n_classes
        return (cost, label_score, label_score_global, predicted_classes,
                score_function, beta_class, n_yi_s)

    def _compute_dist(self, cost, y):
        """Compute the sample distribution (i.e. the weights to use)."""
        n_samples = y.shape[0]
        # dist is forced to be c-contiguous so that sub-arrays of dist used
        # as weights for the weak classifiers are also c-contiguous, which is
        # required by some scikit-learn classifiers (for example
        # sklearn.svm.SVC)
        dist = np.empty(cost.shape[:2], dtype=cost.dtype, order="C")
        # NOTE: In Sokol's PhD thesis, the formula for dist is mistakenly given
        # with a minus sign in section 2.2.2 page 31
        dist[:, :] = cost[:, np.arange(n_samples), y] \
            / np.sum(cost[:, np.arange(n_samples), y], axis=1)[:, np.newaxis]
        return dist

    def _indicatrice(self, predicted_classes, y_i):
        n_samples = y_i.shape[0]
        indicate_ones = np.zeros((self.n_views_, n_samples, self.n_classes_), dtype=np.int)
        indicatrice_one_yi = np.zeros((self.n_views_, n_samples, self.n_classes_), dtype=np.int)
        indicate_ones[np.arange(self.n_views_)[:, np.newaxis],
                    np.arange(n_samples)[np.newaxis, :],
                    predicted_classes[np.arange(self.n_views_), :]] = 1
        indicate_ones[:, np.arange(n_samples), y_i] = 0
        indicatrice_one_yi[:, np.arange(n_samples), y_i] = 1
        delta = np.ones((self.n_views_, n_samples, self.n_classes_), dtype=np.int)
        delta[:, np.arange(n_samples), y_i] = -1
        # indic_minus_one = np.where(np.arange(self.n_classes_) == y)
        return indicate_ones, indicatrice_one_yi, delta

    def _compute_edges(self, cost, predicted_classes, y):
        """Compute edge values for the cost matrices for all the views."""
        n_views = predicted_classes.shape[0]
        n_samples = y.shape[0]
        edges = - np.sum(
            cost[np.arange(n_views)[:, np.newaxis],
                 np.arange(n_samples)[np.newaxis, :],
                 predicted_classes[np.arange(n_views), :]], axis=1) \
            / (np.sum(cost, axis=(1, 2))
               - np.sum(cost[:, np.arange(n_samples), y], axis=1))
        return edges

    def _compute_alphas(self, edges):
        """Compute values of confidence rate alpha given edge values."""
        np.where(edges > 1.0, edges, 1.0)
        alphas = 0.5 * np.log((1. + edges) / (1. - edges))
        if np.any(np.isinf(alphas)) or np.any(np.isnan(alphas)):

            alphas[np.where(np.isnan(alphas))[0]] = 1.0
            alphas[np.where(np.isinf(alphas))[0]] = 1.0
        return alphas

    def _compute_cost(self, label_score, predicted_classes, y, alphas, betas,
                      use_coop_coef=True):
        """Update label_score and compute the cost matrices for all views."""
        # use_coop_coef is a boolean parameter used to choose if the
        # cooperation coefficients are computed and taken into account when
        # updating the cost matrices.
        # It is introduced here for future explorations.
        n_views = predicted_classes.shape[0]
        n_samples = y.shape[0]
        if use_coop_coef:
            # coop_coef = self._compute_coop_coef(predicted_classes, y)

            # ajout mucumbo verifier les dim
            # ????? coop_cof_beta = betas[predicted_classes]
            increment = alphas[:, np.newaxis, np.newaxis] * betas[:, np.newaxis, :]
            increment = np.tile(increment,(1, n_samples, 1))
        else:
            increment = np.tile(alphas[:, np.newaxis, np.newaxis], (1, n_samples, self.n_classes_))
        label_score[np.arange(n_views)[:, np.newaxis],
                    np.arange(n_samples)[np.newaxis, :],
                    predicted_classes[np.arange(n_views), :]] += increment[np.arange(n_views)[:, np.newaxis],
                                                                           np.arange(n_samples)[np.newaxis, :] ,
                                                                           predicted_classes[np.arange(n_views), :]]
        cost = np.exp(
            label_score
            - label_score[:, np.arange(n_samples), y][:, :, np.newaxis]) / self.n_yi[np.newaxis, np.newaxis, :]
        score_function_dif = np.exp(
            label_score
            - label_score[:, np.arange(n_samples), y][:, :, np.newaxis]) / self.n_yi[np.newaxis, np.newaxis, :]
        cost[:, np.arange(n_samples), y] -= np.sum(cost, axis=2)
        return (cost, label_score, score_function_dif)

    def _prepare_beta_solver(self):
        view = self.n_views_
        m = self.n_classes_
        A = matrix(0.0, (view, m * view))
        one_vector = np.ones((m))
        for v in range(view):
            A[v, v*m : (v*m) +m] = 1
        b = matrix(1.0, (view,1))
        l={'l': 2*view*m}
        G = matrix(0.0, (2*m * view, m * view))
        one_diag_matrix = matrix(1.0, (m*view,1))
        G_1 = spdiag(one_diag_matrix)
        G[0:m * view, :] = G_1
        G[m* view:2* m * view, :] = -1.0* G_1
        h = matrix(0.0, (2*m*view,1))
        h[0:m*view] = 1.0
        return A, b, G, h, l

    def _compute_betas(self, alphas, y, score_function_dif_Tminus1, predicted_classes):
        """
        minimization of
        :math:` argmin on /beta_{t,c} sum_{v,i,c!=y_i}{frac{1}{n_y_i} cost_{t-1} exp{/apha_{v} \beta_{c}^{b}'

        Parameters
        ----------
        edges : array-like
        alphas
        y
        estimators

        Returns
        -------
        betas arrays
        """
        # delta = self.delta_c_yi(predicted_classes, y)
        indicat, indicate_yi, delta = self._indicatrice(predicted_classes, y)
        delta_vue = np.block(np.split(delta, self.n_views_, axis=0)).squeeze()
        indicate_vue = np.block(np.split(indicat, self.n_views_, axis=0)).squeeze()
        indicate_vue_yi = np.block(np.split(indicate_yi, self.n_views_, axis=0)).squeeze()
        score_function_Tminus1_vue = np.block(np.split(score_function_dif_Tminus1, self.n_views_, axis=0)).squeeze()
        A, b, G, h, l = self._prepare_beta_solver()
        solver = self._solver_cp_forbeta(alphas, indicate_vue, indicate_vue_yi, delta_vue, score_function_Tminus1_vue, A, b, G, h, l)
        betas = np.array(solver)
        betas = betas.reshape((self.n_views_, self.n_classes_))
        return betas

    def _solver_cp_forbeta(self, alphas, indicate_vue, indicate_vue_yi, delta_vue, score_function_dif_Tminus1, A, b, G, h, l):
        solvers.options['show_progress'] = False
        n_view = self.n_views_
        m = self.n_classes_
        coef = 1.0/np.tile(self.n_yi, self.n_views_).squeeze() * score_function_dif_Tminus1
        zeta_v =  np.repeat(alphas, self.n_classes_) * indicate_vue * delta_vue
        zeta_v_yi = np.repeat(alphas, self.n_classes_) * indicate_vue_yi * delta_vue
        zeta = zeta_v + zeta_v_yi
        zeta2 = zeta**2
        def F(x=None, z=None):
            if x is None:
                # l'algorithme fonctionne de manière itérative
                # il faut choisir un x initial, c'est ce qu'on fait ici
                return 0, matrix(1.0, (n_view*m, 1))
            if min(x) < 0.0:
                return None   # cas impossible
            # ici commence le code qui définit ce qu'est une itération
            f = sum(matrix(coef * exp( matrix(zeta * x.T))   ))
            Df = matrix(np.sum( zeta * coef * exp(matrix( zeta * x.T ) ), axis=0 )).T # -(x**-1).T
            if z is None: return f, Df
            H = spdiag(z[0] * matrix(np.sum(coef * zeta2 * exp( matrix(zeta* x.T) ), axis= 0))) ## beta**(-2))
            return f, Df, H
        try:
            solver = solvers.cp(F, A=A, b=b, G=G, h=h, dim={'l':2*n_view*m})['x']
        except ValueError or ArithmeticError or OverflowError as e:
            norm = np.sum(1.0/self.n_yi)
            yi_norm = self.n_yi * (norm )
            solver = matrix(1.0/np.tile(yi_norm, n_view).squeeze(), (n_view * m, 1))
            print("Value Error on the evaluation on beta coefficient %s "% e)
        return solver

    def _compute_predictions(self, X):
        """Compute predictions for all the stored estimators on the data X."""
        n_samples = X.shape[0]
        n_estimators = len(self.estimators_)
        predictions = np.zeros((n_samples, n_estimators), dtype=np.int64)
        for ind_estimator, estimator in enumerate(self.estimators_):
            # no best view in mucumbo but all view
            # ind_view = self.best_views_[ind_estimator]
            ind_view = ind_estimator % self.n_views_
            predictions[:, ind_estimator] \
                = estimator.predict(X._extract_view(ind_view))
        return predictions

    def fit(self, X, y, views_ind=None):
        """Build a multimodal boosted classifier from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = (n_samples, n_features)
            Training multi-view input samples.
            Sparse matrix can be CSC, CSR, COO, DOK, or LIL.
            COO, DOK and LIL are converted to CSR.

        y : array-like, shape = (n_samples,)
            Target values (class labels).

        views_ind : array-like (default=[0, n_features//2, n_features])
            Paramater specifying how to extract the data views from X:

            - If views_ind is a 1-D array of sorted integers, the entries
              indicate the limits of the slices used to extract the views,
              where view ``n`` is given by
              ``X[:, views_ind[n]:views_ind[n+1]]``.

              With this convention each view is therefore a view (in the NumPy
              sense) of X and no copy of the data is done.

            - If views_ind is an array of arrays of integers, then each array
              of integers ``views_ind[n]`` specifies the indices of the view
              ``n``, which is then given by ``X[:, views_ind[n]]``.

              With this convention each view creates therefore a partial copy
              of the data in X. This convention is thus more flexible but less
              efficient than the previous one.

        Returns
        -------
        self : object
            Returns self.

        Raises
        ------
        ValueError  estimator must support sample_weight

        ValueError where `X` and `view_ind` are not compatibles
        """
        warnings.filterwarnings("ignore")
        if (self.base_estimator is None or
                isinstance(self.base_estimator, (BaseDecisionTree,
                                                 BaseForest))):
            dtype = DTYPE
            accept_sparse = 'csc'
        else:
            dtype = None
            accept_sparse = ['csr', 'csc']
        if views_ind is None:
            if X.shape[1] > 1:
                views_ind = np.array([0, X.shape[1]//2, X.shape[1]])
            else:
                views_ind = np.array([0, X.shape[1]])

        self.X_ = self._global_X_transform(X, views_ind=views_ind)
        views_ind_, n_views = self.X_._validate_views_ind(views_ind,
                                                          X.shape[1])
        check_X_y(self.X_, y, accept_sparse=accept_sparse, dtype=dtype)
        check_classification_targets(y)
        self._validate_estimator()

        self.n_iterations_ = self.n_estimators // n_views
        self.classes_, y = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        self.n_views_ = n_views
        self.n_features_ = X.shape[1]
        if self.n_classes_ == 1:
            # This case would lead to division by 0 when computing the cost
            # matrix so it needs special handling (but it is an obvious case as
            # there is only one single class in the data).
            self.estimators_ = []
            self.estimator_weights_alpha_ = np.array([], dtype=np.float64)
            self.estimator_weights_beta_ = np.zeros((self.n_iterations_, n_views), dtype=np.float)
            self.estimator_errors_ = np.array([], dtype=np.float64)
            return
        # probablement la list de  h de t global  que l'on a a la fin
        self.estimators_ = []
        # modification mu cumbo
        # mettre deux dim sur n_estimators * n_views
        self.estimator_weights_alpha_ = np.zeros((self.n_iterations_, n_views), dtype=np.float64)
        self.estimator_weights_beta_ = np.zeros((self.n_iterations_, n_views, self.n_classes_), dtype=np.float)
        self.estimator_errors_ = np.zeros((n_views, self.n_iterations_), dtype=np.float64)

        random_state = check_random_state(self.random_state)
        (cost, label_score, label_score_global,
         predicted_classes, score_function_dif, betas, n_yi) = self._init_var(n_views, y)
        self.n_yi = n_yi
        for current_iteration in range(self.n_iterations_):
            # list de h pris a l'etape t
            dist = self._compute_dist(cost, y)
            # get h_t _i  with edges delta
            for ind_view in range(n_views):
                estimator = self._make_estimator(append=False,
                                                 random_state=random_state)
                estimator.fit(self.X_._extract_view(ind_view), y,
                              sample_weight=dist[ind_view, :])
                predicted_classes[ind_view, :] = estimator.predict(
                    self.X_._extract_view(ind_view))
                self.estimators_.append(estimator)

            # end of choose cost matrix
            #   TO DO estimator_errors_ estimate
            ###########################################


            #############self.estimator_errors_[current_iteration] = to do
            # update C_t de g

            edges = self._compute_edges(cost, predicted_classes, y)
            alphas = self._compute_alphas(edges)
            # modif mu cumbo
            self.estimator_weights_alpha_[current_iteration, :] = alphas

            betas = self._compute_betas(alphas, y, score_function_dif, predicted_classes)
            self.estimator_weights_beta_[current_iteration, :, :] = betas
            # update cost matrices C_t_j ...
            cost, label_score, score_function_dif = self._compute_cost(
                label_score, predicted_classes, y, alphas, betas, True)
        return self

    def decision_function(self, X):
        """Compute the decision function of X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = (n_samples, n_features)
            Multi-view input samples.
            Sparse matrix can be CSC, CSR, COO, DOK, or LIL.
            COO, DOK and LIL are converted to CSR.

        Returns
        -------
        dec_fun : numpy.ndarray, shape = (n_view, n_samples, k)
            Decision function of the input samples.
            The order of outputs is the same of that of the `classes_`
            attribute.
            Binary classification is a special cases with ``k == 1``,
            otherwise ``k == n_classes``. For binary classification,
            values <=0 mean classification in the first class in ``classes_``
            and values >0 mean classification in the second class in
            ``classes_``.
        """
        check_is_fitted(self, ("estimators_", "estimator_weights_alpha_","n_views_",
                               "estimator_weights_beta_", "n_classes_", "X_"))
        X = self._global_X_transform(X, views_ind=self.X_.views_ind)
        X = self._validate_X_predict(X)

        n_samples = X.shape[0]
        n_estimators = len(self.estimators_)
        n_classes = self.n_classes_
        n_iterations = self.n_iterations_
        predictions = self._compute_predictions(X)
        n_views = self.n_views_

        dec_func = np.zeros((n_samples, n_classes))
        # update muCombo
        # for ind_estimator in range(n_estimators):
        for ind_estimator in range(n_estimators):
            ind_iteration = ind_estimator // self.n_views_
            current_vue = ind_estimator % self.n_views_
            vector_classes = predictions[:, ind_estimator]
            dec_func[np.arange(n_samples), vector_classes] \
                += (self.estimator_weights_alpha_[ind_iteration, current_vue, np.newaxis] * \
                   self.estimator_weights_beta_[ind_iteration, current_vue,  vector_classes])

        if n_classes == 2:
            dec_func[:, 0] *= -1
            return np.sum(dec_func, axis=1)

        return dec_func

    def staged_decision_function(self, X):
        """Compute decision function of X for each boosting iteration.

        This method allows monitoring (i.e. determine error on testing set)
        after each boosting iteration.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = (n_samples, n_features)
            Multi-view input samples.
            Sparse matrix can be CSC, CSR, COO, DOK, or LIL.
            COO, DOK and LIL are converted to CSR.

        Returns
        -------
        dec_fun : generator of numpy.ndarrays, shape = (n_samples, k)
            Decision function of the input samples.
            The order of outputs is the same of that of the `classes_`
            attribute.
            Binary classification is a special cases with ``k == 1``,
            otherwise ``k==n_classes``. For binary classification,
            values <=0 mean classification in the first class in ``classes_``
            and values >0 mean classification in the second class in
            ``classes_``.
        """
        check_is_fitted(self, ("estimators_", "estimator_weights_alpha_","n_views_",
                               "estimator_weights_beta_", "n_classes_"))
        X = self._global_X_transform(X, views_ind=self.X_.views_ind)
        X = self._validate_X_predict(X)

        n_samples = X.shape[0]
        n_stage = len(self.estimators_)
        n_classes = self.n_classes_
        n_views = self.n_views_
        predictions = self._compute_predictions(X)

        dec_func = np.zeros((n_samples, n_classes))
        for ind_e in range(n_stage):
            vector_classes = predictions[:, ind_e]
            current_vue = ind_e % self.n_views_
            ind_iteration = ind_e // self.n_views_
            dec_func[np.arange(n_samples), vector_classes] \
                += (self.estimator_weights_alpha_[ind_iteration, current_vue, np.newaxis] * \
                   self.estimator_weights_beta_[ind_iteration, current_vue,  vector_classes])
            if n_classes == 2:
                tmp_dec_func = np.array(dec_func)
                tmp_dec_func[ :, 0] *= -1
                yield tmp_dec_func.sum(axis=1)

            else:
                yield np.array(dec_func)

    def predict(self, X):
        """Predict classes for X.

        The predicted class of an input sample is computed as the weighted mean
        prediction of the classifiers in the ensemble.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = (n_samples, n_features)
            Multi-view input samples.
            Sparse matrix can be CSC, CSR, COO, DOK, or LIL.
            COO, DOK and LIL are converted to CSR.

        Returns
        -------
        y : numpy.ndarray, shape = (n_samples,)
            Predicted classes.

        Raises
        ------
        ValueError   'X' input matrix must be have the same total number of features
                     of 'X' fit data
        """
        pred = self.decision_function(X)

        if self.n_classes_ == 2:
            return self.classes_.take(pred > 0, axis=0)

        return self.classes_.take(np.argmax(pred, axis=1), axis=0)

    def staged_predict(self, X):
        """Return staged predictions for X.

        The predicted class of an input sample is computed as the weighted mean
        prediction of the classifiers in the ensemble.

        This generator method yields the ensemble prediction after each
        iteration of boosting and therefore allows monitoring, such as to
        determine the prediction on a test set after each boost.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = (n_samples, n_features)
            Multi-view input samples.
            Sparse matrix can be CSC, CSR, COO, DOK, or LIL.
            COO, DOK and LIL are converted to CSR.

        Returns
        -------
        y : generator of numpy.ndarrays, shape = (n_samples,)
            Predicted classes.
        """

        n_classes = self.n_classes_
        classes = self.classes_
        X = self._validate_X_predict(X)
        if n_classes == 2:
            for pred in self.staged_decision_function(X):
                yield np.array(classes.take(pred > 0, axis=0))
        else:
            for pred in self.staged_decision_function(X):
                yield np.array(classes.take(np.argmax(pred, axis=1), axis=0))

    def score(self, X, y):
        """Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = (n_samples, n_features)
            Multi-view test samples.
            Sparse matrix can be CSC, CSR, COO, DOK, or LIL.
            COO, DOK and LIL are converted to CSR.
        y : array-like, shape = (n_samples,)
            True labels for X.

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        X = self._validate_X_predict(X)
        return super(MuCumboClassifier, self).score(X, y)

    def staged_score(self, X, y):
        """Return staged mean accuracy on the given test data and labels.

        This generator method yields the ensemble score after each iteration of
        boosting and therefore allows monitoring, such as to determine the
        score on a test set after each boost.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = (n_samples, n_features)
            Multi-view test samples.
            Sparse matrix can be CSC, CSR, COO, DOK, or LIL.
            COO, DOK and LIL are converted to CSR.
        y : array-like, shape = (n_samples,)
            True labels for X.

        Returns
        -------
        score : generator of floats
            Mean accuracy of self.staged_predict(X) wrt. y.
        """
        for y_pred in self.staged_predict(X):
            yield accuracy_score(y, y_pred)
