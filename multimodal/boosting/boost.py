import numpy as np
from abc import ABCMeta
from sklearn.utils import check_array, check_X_y, check_random_state
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree.tree import BaseDecisionTree
from sklearn.tree._tree import DTYPE
from sklearn.ensemble.forest import BaseForest
from multimodal.datasets.data_sample import DataSample, MultiModalArray

class UBoosting(metaclass=ABCMeta):
    """
    Abstract class MuCumboClassifier and  MumboClassifier should inherit from
    UBoosting for methods
    """

    def _validate_X_predict(self, X):
        """Ensure that X is in the proper format."""
        if (self.base_estimator is None or
                isinstance(self.base_estimator,
                           (BaseDecisionTree, BaseForest))):
            check_array(X, accept_sparse='csr', dtype=DTYPE)

        else:
            check_array(X, accept_sparse=['csr', 'csc'])
        if X.shape[1] != self.n_features_:
            raise ValueError("X doesn't contain the right number of features.")
        return X


    def _validate_views_ind(self, views_ind, n_features):
        """Ensure proper format for views_ind and return number of views."""
        views_ind = np.array(views_ind)
        if np.issubdtype(views_ind.dtype, np.integer) and views_ind.ndim == 1:
            if np.any(views_ind[:-1] >= views_ind[1:]):
                raise ValueError("Values in views_ind must be sorted.")
            if views_ind[0] < 0 or views_ind[-1] > n_features:
                raise ValueError("Values in views_ind are not in a correct "
                                 + "range for the provided data.")
            self.view_mode_ = "slices"
            n_views = views_ind.shape[0]-1
        else:
            if views_ind.ndim == 1:
                if not views_ind.dtype == np.object:
                    raise ValueError("The format of views_ind is not "
                                     + "supported.")
                for ind, val in enumerate(views_ind):
                    views_ind[ind] = np.array(val)
                    if not np.issubdtype(views_ind[ind].dtype, np.integer):
                        raise ValueError("Values in views_ind must be "
                                         + "integers.")
                    if views_ind[ind].min() < 0 \
                            or views_ind[ind].max() >= n_features:
                        raise ValueError("Values in views_ind are not in a "
                                         + "correct range for the provided "
                                         + "data.")
            elif views_ind.ndim == 2:
                if not np.issubdtype(views_ind.dtype, np.integer):
                    raise ValueError("Values in views_ind must be integers.")
                if views_ind.min() < 0 or views_ind.max() >= n_features:
                    raise ValueError("Values in views_ind are not in a "
                                     + "correct range for the provided data.")
            else:
                raise ValueError("The format of views_ind is not supported.")
            self.view_mode_ = "indices"
            n_views = views_ind.shape[0]
        return (views_ind, n_views)

    def _global_X_transform(self, X, views_ind=None):
        X_ = None
        if isinstance(X, np.ndarray) and X.ndim == 1:
            X_= MultiModalArray(X, views_ind)
        elif isinstance(X, dict):
            X_= MultiModalArray(X)
        elif isinstance(X, np.ndarray) and X.ndim > 1:
            X_ = MultiModalArray(X, views_ind)
        if not isinstance(X_, MultiModalArray):
            raise TypeError("Input format is not reconized")
        if hasattr(self, "X_"):
            if not self.X_.viexs_ind == views_ind:
                raise ValueError("Input format (viewd, features) for fit and predict must be the same")
        return X_