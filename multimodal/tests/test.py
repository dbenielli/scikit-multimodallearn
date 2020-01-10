
from abc import ABCMeta
import numpy as np
import numpy.ma as ma
import scipy.sparse as sp

from multimodal.boosting.mumbo import MumboClassifier

class MultiModalData(metaclass=ABCMeta):

    @staticmethod
    def _first_validate_views_ind(views_ind, n_features):
        """Ensure proper format for views_ind and return number of views."""
        views_ind = np.array(views_ind)
        if np.issubdtype(views_ind.dtype, np.integer) and views_ind.ndim == 1:
            if np.any(views_ind[:-1] >= views_ind[1:]):
                raise ValueError("Values in views_ind must be sorted.")
            if views_ind[0] < 0 or views_ind[-1] > n_features:
                raise ValueError("Values in views_ind are not in a correct "
                                 + "range for the provided data.")
            view_mode_ = "slices"
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
            view_mode_ = "indices"
            n_views = views_ind.shape[0]
        return (views_ind, n_views, view_mode_)

    def _extract_view(self, ind_view):
        """Extract the view for the given index ind_view from the dataset X."""
        if self.view_mode_ == "indices":
            return self[:, self.views_ind[ind_view]]
        else:
            return self[:, self.views_ind[ind_view]:self.views_ind[ind_view+1]]

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
        self.views_ind = views_ind
        self.n_views = n_views
        return (views_ind, n_views)

class MultiModalSparseInfo():

    def __init__(self, data, view_ind=None):
        """Constructor of Metriclearn_array"""
        shapes_int = []
        index = 0
        new_data = np.ndarray([])
        n_views = data.size
        thekeys = None
        # view_ind_self =  None
        view_mode = 'slices'

        if (sp.issparse(data)) and data.ndim > 1:
            if  view_ind is not None:
                try:
                    view_ind = np.asarray(view_ind)
                except :
                    raise TypeError("n_views should be list or nparray")
            elif view_ind is None:
                if data.shape[1] > 1:
                    view_ind = np.array([0, data.shape[1]//2, data.shape[1]])
                else:
                    view_ind = np.array([0, data.shape[1]])

            new_data = data
            # view_ind_self = view_ind
        view_ind, n_views, view_mode = self._first_validate_views_ind(view_ind,
                                                                      data.shape[1])
        if view_ind.ndim == 1 and view_mode.startswith("slicing"):
            shapes_int = [in2 - in1 for in1, in2 in zip(view_ind, view_ind[1:])]

        if data.shape[0] < 1 or data.shape[1] < 1:
            raise ValueError("input data shouldbe not empty")
        self.view_mode_ = view_mode
        self.views_ind = view_ind
        self.shapes_int = shapes_int
        self.n_views = n_views


class MultiModalSparseArray(sp.csr_matrix, sp.csc_matrix, MultiModalSparseInfo, MultiModalData):
    """
    MultiModalArray inherit from numpy ndarray


    Parameters
    ----------

    data : can be
             - dictionary of multiview array with shape = (n_samples, n_features)  for multi-view
                  for each view.
               {0: array([[]],
                1: array([[]],
                ...}
             - numpy array like with shape = (n_samples, n_features)  for multi-view
                  for each view.
                [[[...]],
                 [[...]],
                 ...]
             - {array like} with (n_samples, nviews *  n_features) with 'views_ind' diferent to 'None'
                for Multi-view input samples.




        views_ind : array-like (default= None ) if None
                    [0, n_features//2, n_features]) is constructed (2 views)
                    Paramater specifying how to extract the data views from X:

            - views_ind is a 1-D array of sorted integers, the entries
              indicate the limits of the slices used to extract the views,
              where view ``n`` is given by
              ``X[:, views_ind[n]:views_ind[n+1]]``.

        Attributes
        ----------

        view_ind : list of views' indice  (may be None)

        n_views : int number of views

        shapes_int: list of int numbers of feature for each views

        keys : name of key, where data come from a dictionary


    :Example:

    >>> from multimodal.datasets.base import load_dict
    >>> from multimodal.tests.datasets.get_dataset_path import get_dataset_path
    >>> from multimodal.datasets.data_sample import DataSample
    >>> file = 'input_x_dic.pkl'
    >>> data = load_dict(get_dataset_path(file))

    """

    def __init__(self, *arg, **kwargs ):
        """Constructor of Metriclearn_array"""
        if sp.issparse(arg[0]):
            MultiModalSparseInfo.__init__(self, *arg)
            if isinstance(arg[0], sp.csr_matrix) :
                sp.csr_matrix.__init__(self, arg[0])
            elif isinstance(arg[0], sp.csc_matrix):
                sp.csc_matrix.__init__(self, arg[0])
            else:
                raise TypeError("This sparse format is not supported")
        else:
            if isinstance(self,sp.csr_matrix):
               sp.csr_matrix.__init__(self, *arg, **kwargs)
            elif isinstance(self, sp.csc_matrix):
               sp.csc_matrix.__init__(self, *arg, **kwargs)




if __name__ == '__main__':
    rng = np.random.RandomState(0)
    X = rng.rand(40, 10)
    X[X < .8] = 0
    X_csr = sp.csr_matrix(X)
    y = (4 * rng.rand(40)).astype(np.int)
    X_ = MultiModalSparseArray(X_csr)
    print(X_.shape)
    print(X_[:,0:1])

    X = np.array([[3, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 1]])
    y = [1, 1, 1, 2, 2, 2]
    clf =  MumboClassifier()
    clf.fit(X, y)