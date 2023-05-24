# -*- coding: utf-8 -*-
# ######### COPYRIGHT #########
#
# Copyright(c) 2020
# -----------------
#
# * Université d'Aix Marseille (AMU) -
# * Centre National de la Recherche Scientifique (CNRS) -
# * Université de Toulon (UTLN).
# * Copyright © 2019-2020 AMU, CNRS, UTLN
#
# Contributors:
# ------------
#
# * Sokol Koço <sokol.koco_AT_lis-lab.fr>
# * Cécile Capponi <cecile.capponi_AT_univ-amu.fr>
# * Florent Jaillet <florent.jaillet_AT_math.cnrs.fr>
# * Dominique Benielli <dominique.benielli_AT_univ-amu.fr>
# * Riikka Huusari <rikka.huusari_AT_univ-amu.fr>
# * Baptiste Bauvin <baptiste.bauvin_AT_univ-amu.fr>
# * Hachem Kadri <hachem.kadri_AT_lis-lab.fr>
#
# Description:
# -----------
#
# The multimodal package implement classifiers multiview, 
# MumboClassifier class, MuComboClassifier class, MVML class, MKL class.
# compatible with sklearn
#
# Version:
# -------
#
# * multimodal version = 0.0.dev0
#
# Licence:
# -------
#
# License: New BSD License
#
#
# ######### COPYRIGHT #########
r"""Multimodal Boosting

This module contains a **Mu**\ lti\ **M**\ odal **Bo**\ osting (**MuMBo**)
estimator for classification implemented in the ``MumboClassifier`` class.
"""
import numpy as np

from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble import BaseEnsemble
from sklearn.ensemble._base import _set_random_states
from sklearn.ensemble._forest import BaseForest
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import BaseDecisionTree
from sklearn.tree._tree import DTYPE
from sklearn.utils import check_array, check_X_y, check_random_state
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, has_fit_parameter
from .boost import UBoosting


class MumboClassifier(BaseEnsemble, ClassifierMixin, UBoosting):
    r"""It then iterates the process on the same dataset but where the weights of
    incorrectly classified instances are adjusted such that subsequent
    classifiers focus more on difficult cases.
    A MuMBo classifier.

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
        and `n_classes_` attributes. The default is a DecisionTreeClassifie  
        with parameter ``max_depth=1``.

    n_estimators : integer, optional (default=50)
        Maximum number of estimators at which boosting is terminated.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    best_view_mode : {"edge", "error"}, optional (default="edge")
        Mode used to select the best view at each iteration:

        - if ``best_view_mode == "edge"``, the best view is the view maximizing
          the edge value (variable δ (*delta*) in [1]_),
        - if ``best_view_mode == "error"``, the best view is the view
          minimizing the classification error.

    Attributes
    ----------
    estimators\_ : list of classifiers
        Collection of fitted sub-estimators.

    classes\_ : numpy.ndarray, shape = (n_classes,)
        Classes labels.

    n_classes\_ : int
        Number of classes.

    estimator_weights\_ : numpy.ndarray of floats, shape = (len(estimators\  
        Weights for each estimator in the boosted ensemble.

    estimator_errors_ : array of floats
        Empirical loss for each iteration.


    best_views\_ : numpy.ndarray of integers, shape = (len(estimators\_),)
        Indices of the best view for each estimator in the boosted ensemble.

    Examples
    --------
    >>> from multimodal.boosting.mumbo import MumboClassifier
    >>> from sklearn.datasets import load_iris
    >>> X, y = load_iris(return_X_y=True)
    >>> views_ind = [0, 2, 4]  # view 0: sepal data, view 1: petal data
    >>> clf = MumboClassifier(random_state=0)
    >>> clf.fit(X, y, views_ind)  # doctest: +NORMALIZE_WHITESPACE
    MumboClassifier(random_state=0)
    >>> print(clf.predict([[ 5.,  3.,  1.,  1.]]))
    [1]
    >>> views_ind = [[0, 2], [1, 3]]  # view 0: length data, view 1: width data
    >>> clf = MumboClassifier(random_state=0)
    >>> clf.fit(X, y, views_ind)  # doctest: +NORMALIZE_WHITESPACE
    MumboClassifier(random_state=0)
    >>> print(clf.predict([[ 5.,  3.,  1.,  1.]]))
    [1]

    >>> from sklearn.tree import DecisionTreeClassifier
    >>> base_estimator = DecisionTreeClassifier(max_depth=2)
    >>> clf = MumboClassifier(base_estimator=base_estimator, random_state=0)
    >>> clf.fit(X, y, views_ind)  # doctest: +NORMALIZE_WHITESPACE  
    MumboClassifier(base_estimator=DecisionTreeClassifier(max_depth=2),
                    random_state=0)
    >>> print(clf.predict([[ 5.,  3.,  1.,  1.]]))
    [1]

    See also
    --------
    sklearn.ensemble.AdaBoostClassifier,
    sklearn.ensemble.GradientBoostingClassifier,
    sklearn.tree.DecisionTreeClassifier

    References
    ----------
    .. [1] Sokol Koço,
           "Tackling the uneven views problem with cooperation based ensemble
           learning methods", 
    """

    def __init__(self,
                 base_estimator=None,
                 n_estimators=50,
                 random_state=None,
                 best_view_mode="edge"):

        if type(base_estimator) is list:
            self.base_estimator = base_estimator
            self.n_estimators = n_estimators
            self.estimator_params = [tuple() for _ in base_estimator]

        else:
            super(MumboClassifier, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators)

        self.random_state = random_state
        self.best_view_mode = self._validate_best_view_mode(best_view_mode)

    def _validate_estimator(self):
        """Check the estimator and set the base_estimator_ attribute."""
        super(MumboClassifier, self)._validate_estimator(
            default=DecisionTreeClassifier(max_depth=1))
        if type(self.base_estimator_) is list:
            for estimator in self.base_estimator_:
                if not has_fit_parameter(estimator, "sample_weight"):
                    raise ValueError("%s doesn't support sample_weight."
                                     % estimator.__class__.__name__)

        else:
            if not has_fit_parameter(self.base_estimator_, "sample_weight"):
                raise ValueError("%s doesn't support sample_weight."
                                 % self.base_estimator_.__class__.__name__)

    def _make_estimator(self, append=True, random_state=None, ind_view=0):
        if type(self.base_estimator_) is list:
            estimator = clone(self.base_estimator_[ind_view])
            estimator.set_params(**{p: getattr(self, p)
                                    for p in self.estimator_params[ind_view]})
            # TODO : modify estimator_params to be able to set a list

            if random_state is not None:
                _set_random_states(estimator, random_state)

            if append:
                self.estimators_.append(estimator)

            return estimator

        else:
            return super(MumboClassifier, self)._make_estimator(append=append,
                                                         random_state=random_state)


    def _validate_best_view_mode(self, best_view_mode):
        """Ensure that best_view_mode has a proper value."""
        if best_view_mode not in ("edge", "error"):
            raise ValueError('best_view_mode value must be either "edge" '
                             + 'or "error"')
        return best_view_mode

    def _init_var(self, n_views, y):
        "Create and initialize the variables used by the MuMBo algorithm."
        n_classes = self.n_classes_
        n_samples = y.shape[0]

        cost = np.ones((n_views, n_samples, n_classes))
        cost[:, np.arange(n_samples), y] = -(n_classes-1)

        cost_global = np.ones((n_samples, n_classes))
        cost_global[np.arange(n_samples), y] = -(n_classes-1)

        label_score = np.zeros((n_views, n_samples, n_classes))

        label_score_global = np.zeros((n_samples, n_classes))

        predicted_classes = np.empty((n_views, n_samples), dtype=np.int64)

        return (cost, cost_global, label_score, label_score_global,
                predicted_classes)

    def _compute_edge_global(self, cost_global, predicted_classes, y):
        """Compute edge values for the global cost matrix."""
        n_samples = y.shape[0]
        edge_global = - np.sum(
            cost_global[np.arange(n_samples), predicted_classes], axis=1) \
            / (np.sum(cost_global)
               - np.sum(cost_global[np.arange(n_samples), y]))
        return edge_global

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

    def _compute_coop_coef(self, predicted_classes, y):
        """Compute the cooperation coefficients."""
        coop_coef = np.zeros(predicted_classes.shape)
        coop_coef[predicted_classes == y] = 1.
        coop_coef[:, np.logical_not(coop_coef.any(axis=0))] = 1.
        return coop_coef

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
        if np.any(np.isinf(alphas)):
            if isinstance(alphas, float):
                alphas = 1.0
            else:
                alphas[np.where(np.isinf(alphas))[0]] = 1.0
        if np.any(np.isnan(alphas)):
            if isinstance(alphas, float):
                alphas = 1.0
            else:
                alphas[np.where(np.isnan(alphas))[0]] = 1.0
        return alphas

    def _compute_cost_global(self, label_score_global, best_predicted_classes,
                             y, alpha):
        """Update label_score_global and compute the global cost matrix."""
        n_samples = y.shape[0]
        label_score_global[np.arange(n_samples), best_predicted_classes] \
            += alpha
        cost_global = np.exp(
            label_score_global
            - label_score_global[np.arange(n_samples), y][:, np.newaxis])
        cost_global[np.arange(n_samples), y] -= np.sum(cost_global, axis=1)
        return (cost_global, label_score_global)

    def _compute_cost(self, label_score, predicted_classes, y, alphas,
                      use_coop_coef=True):
        """Update label_score and compute the cost matrices for all views."""
        # use_coop_coef is a boolean parameter used to choose if the
        # cooperation coefficients are computed and taken into account when
        # updating the cost matrices.
        # It is introduced here for future explorations.
        n_views = predicted_classes.shape[0]
        n_samples = y.shape[0]
        if use_coop_coef:
            coop_coef = self._compute_coop_coef(predicted_classes, y)
            increment = alphas[:, np.newaxis] * coop_coef
        else:
            increment = alphas[:, np.newaxis]
        label_score[np.arange(n_views)[:, np.newaxis],
                    np.arange(n_samples)[np.newaxis, :],
                    predicted_classes[np.arange(n_views), :]] += increment
        cost = np.exp(
            label_score
            - label_score[:, np.arange(n_samples), y][:, :, np.newaxis])
        cost[:, np.arange(n_samples), y] -= np.sum(cost, axis=2)
        return (cost, label_score)

    def _compute_predictions(self, X):
        """Compute predictions for all the stored estimators on the data X."""
        n_samples = X.shape[0]
        n_estimators = len(self.estimators_)
        predictions = np.zeros((n_samples, n_estimators), dtype=np.int64)
        for ind_estimator, estimator in enumerate(self.estimators_):
            ind_view = self.best_views_[ind_estimator]
            predictions[:, ind_estimator] \
                = estimator.predict(X._extract_view(ind_view))
        return predictions

    def fit(self, X, y, views_ind=None):
        """Build a multimodal boosted classifier from the training set (X, y).

        Parameters
        ----------
        X : dict dictionary with all views
            or
            `MultiModalData` ,  `MultiModalArray`, `MultiModalSparseArray`
            or
            {array-like, sparse matrix}, shape = (n_samples, n_features)
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
        """
        self.best_view_mode = self._validate_best_view_mode(
            self.best_view_mode)
        if (self.base_estimator is None or
                isinstance(self.base_estimator, (BaseDecisionTree,
                                                 BaseForest))):
            dtype = DTYPE
            accept_sparse = 'csc'
        else:
            dtype = None
            accept_sparse = ['csr', 'csc']

        self.X_ = self._global_X_transform(X, views_ind=views_ind)
        views_ind_, n_views = self.X_._validate_views_ind(self.X_.views_ind,
                                                          self.X_.shape[1])
        check_X_y(self.X_, y, accept_sparse=accept_sparse, dtype=dtype)
        if not isinstance(y, np.ndarray):
            y = np.asarray(y)
        check_classification_targets(y)
        self._validate_estimator()

        self.classes_, y = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        self.n_features_ = self.X_.shape[1]
        self.n_features_in_ = self.n_features_ 
        if self.n_classes_ == 1:
            # This case would lead to division by 0 when computing the cost
            # matrix so it needs special handling (but it is an obvious case as
            # there is only one single class in the data).
            self.estimators_ = []
            self.estimator_weights_ = np.array([], dtype=np.float64)
            self.estimator_errors_ = np.array([], dtype=np.float64)
            self.best_views_ = np.array([], dtype=np.int64)
            return

        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float64)
        self.estimator_errors_ = np.zeros(self.n_estimators, dtype=np.float64)
        self.best_views_ = - np.ones(self.n_estimators, dtype=np.int64)

        random_state = check_random_state(self.random_state)
        (cost, cost_global, label_score, label_score_global,
         predicted_classes) = self._init_var(n_views, y)

        current_iteration = 0
        while True:
            estimators = []
            dist = self._compute_dist(cost, y)
            for ind_view in range(n_views):
                estimator = self._make_estimator(append=False,
                                                 random_state=random_state, ind_view=ind_view)
                estimator.fit(self.X_._extract_view(ind_view), y,
                              sample_weight=dist[ind_view, :])
                estimators.append(estimator)
                predicted_classes[ind_view, :] = estimator.predict(
                    self.X_._extract_view(ind_view))
            edges = self._compute_edge_global(
                cost_global, predicted_classes, y)
            if self.best_view_mode == "edge":
                best_view = np.argmax(edges)
            else:  # self.best_view_mode == "error"
                n_errors = np.sum(predicted_classes != y, axis=1)
                best_view = np.argmin(n_errors)
            edge = edges[best_view]

            if (edge == 1.):
                self.estimator_weights_[0] = 1.
                self.estimator_weights_ = np.resize(self.estimator_weights_, (1, ))
                self.best_views_[0] = best_view
                self.best_views_ = np.resize(self.best_views_, (1, ))
                self.estimators_ = [estimators[best_view]]
                self.estimator_errors_[0] = 0.
                self.estimator_errors_ = np.resize(self.estimator_errors_, (1, ))
                break

            self.estimator_errors_[current_iteration] = (
                    np.average(cost_global[np.arange(y.shape[0]), y])
                    * (-1. / (self.n_classes_-1)))

            alpha = self._compute_alphas(edge)
            self.estimator_weights_[current_iteration] = alpha
            self.best_views_[current_iteration] = best_view
            self.estimators_.append(estimators[best_view])

            if current_iteration == self.n_estimators-1:
                break

            cost_global, label_score_global = self._compute_cost_global(
                label_score_global, predicted_classes[best_view, :], y, alpha)

            edges = self._compute_edges(cost, predicted_classes, y)
            alphas = self._compute_alphas(edges)
            cost, label_score = self._compute_cost(
                label_score, predicted_classes, y, alphas)

            current_iteration += 1

        return self

    def decision_function(self, X):
        """Compute the decision function of X.

        Parameters
        ----------
        X : { array-like, sparse matrix},
            shape = (n_samples, n_views * n_features)
            Multi-view input samples.
            Sparse matrix can be CSC, CSR, COO, DOK, or LIL.
            COO, DOK and LIL are converted to CSR.
            maybe also MultimodalData

        Returns
        -------
        dec_fun : numpy.ndarray, shape = (n_samples, k)
            Decision function of the input samples.
            The order of outputs is the same of that of the `classes_`
            attribute.
            Binary classification is a special cases with ``k == 1``,
            otherwise ``k == n_classes``. For binary classification,
            values <=0 mean classification in the first class in ``classes_``
            and values >0 mean classification in the second class in
            ``classes_``.
        """
        check_is_fitted(self, ("estimators_", "estimator_weights_",
                               "best_views_", "n_classes_", "X_"))
        X = self._global_X_transform(X, views_ind=self.X_.views_ind)
        X = self._validate_X_predict(X)

        n_samples = X.shape[0]
        n_estimators = len(self.estimators_)
        n_classes = self.n_classes_

        predictions = self._compute_predictions(X)

        dec_func = np.zeros((n_samples, n_classes))
        for ind_estimator in range(n_estimators):
            dec_func[np.arange(n_samples), predictions[:, ind_estimator]] \
                += self.estimator_weights_[ind_estimator]

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
            maybe also MultimodalData

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
        check_is_fitted(self, ("estimators_", "estimator_weights_",
                               "n_classes_", "X_"))
        X = self._global_X_transform(X, views_ind=self.X_.views_ind)
        X = self._validate_X_predict(X)

        n_samples = X.shape[0]
        n_estimators = len(self.estimators_)
        n_classes = self.n_classes_

        predictions = self._compute_predictions(X)

        dec_func = np.zeros((n_samples, n_classes))
        for ind_estimator in range(n_estimators):
            dec_func[np.arange(n_samples), predictions[:, ind_estimator]] \
                += self.estimator_weights_[ind_estimator]
            if n_classes == 2:
                tmp_dec_func = np.array(dec_func)
                tmp_dec_func[:, 0] *= -1
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
            Sparse matrix can be CSC, CSR
        y : array-like, shape = (n_samples,)
            True labels for X.

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        return super(MumboClassifier, self).score(X, y)

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
