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
import numpy as np
import scipy.sparse as sp
from abc import ABCMeta
from sklearn.utils import check_array, check_X_y, check_random_state
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import BaseDecisionTree
from sklearn.tree._tree import DTYPE
from sklearn.base import clone
from sklearn.ensemble._base import _set_random_states
from sklearn.ensemble import BaseEnsemble
from sklearn.ensemble._forest import BaseForest
from multimodal.datasets.data_sample import DataSample
from multimodal.datasets.data_sample import MultiModalData, MultiModalArray, MultiModalSparseArray


class UBoosting(metaclass=ABCMeta):
    """
    Abstract class MuComboClassifier and  MumboClassifier should inherit from
    UBoosting for methods
    """

    def _make_unique_estimator(self, base_estimator, estimator_params, append=True, random_state=None, ):
        # Copy/Paste of sklearn.ensebmle.BaseEnsemble._make_estimator
        estimator = clone(base_estimator)
        estimator.set_params(**{p: getattr(self, p)
                                for p in estimator_params})

        if random_state is not None:
            _set_random_states(estimator, random_state)

        if append:
            self.estimators_.append(estimator)

        return estimator

    def _make_estimator(self, append=True, random_state=None, ind_view=0):
        if type(self.base_estimator_) is list:
            return self._make_unique_estimator(self.base_estimator[ind_view], self.estimator_params[ind_view], append=append, random_state=random_state)

        else:
            return self._make_unique_estimator(self.base_estimator, self.estimator_params, append=append, random_state=random_state)

    def _validate_X_predict(self, X):
        """Ensure that X is in the proper format."""
        if X.ndim < 2:
            X = X[np.newaxis, :]
            if X.shape[1] != self.n_features_:
                raise ValueError("Number of features of the model must "
                                    "match the input. Model n_features is %s and "
                                     "input n_features is %s " % (self.n_features_, X.shape[1]))
            else:
                mes = "Reshape your data as a 2D-array "
                raise ValueError(mes)
        if (self.base_estimator is None or
                isinstance(self.base_estimator,
                           (BaseDecisionTree, BaseForest))):
            check_array(X, accept_sparse='csr', dtype=DTYPE)

        else:
            check_array(X, accept_sparse=['csr', 'csc'])
        if X.ndim > 1:
            if X.shape[1] != self.n_features_:
                if X.shape[0] == self.n_features_ and X.shape[1] > 1:
                    raise ValueError("Reshape your data")
                else:
                    raise ValueError("Number of features of the model must "
                                    "match the input. Model n_features is %s and "
                                     "input n_features is %s " % (self.n_features_, X.shape[1]))
        return X

    def _global_X_transform(self, X, views_ind=None):
        if isinstance(X, MultiModalData):
            X_ = X
        elif isinstance(X, sp.spmatrix):
            X_ = MultiModalSparseArray(X, views_ind)
        else:
            X_ = MultiModalArray(X, views_ind)
        return X_
