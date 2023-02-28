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
# MumboClassifier class, MuCumboClassifier class, MVML class, MKL class.
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
#
#
#
#
#

"""This module contains the DataSample class, MultiModalArray, MultiModalSparseArray, MultiModalSparseInfo and MultiModalData, class
The DataSample class encapsulates a sample 's components
nbL and nbEx numbers,
MultiModalArray class inherit from numpy ndarray and contains a 2d data ndarray
with the shape (n_samples, n_view_i * n_features_i)

.. tabularcolumns::  |l|l|l|l|

+----------+------+------+------+
| 0        | 1    | 2    | 3    |
+==========+======+======+======+
| xxxxxxxx | xxxx | xxxx | xxxx |
+----------+------+------+------+
| xxxxxxxx | xxxx | xxxx | xxxx |
+----------+------+------+------+
| xxxxxxxx | xxxx | xxxx | xxxx |
+----------+------+------+------+
| xxxxxxxx | xxxx | xxxx | xxxx |
+----------+------+------+------+
| xxxxxxxx | xxxx | xxxx | xxxx |
+----------+------+------+------+
| xxxxxxxx | xxxx | xxxx | xxxx |
+----------+------+------+------+
| xxxxxxxx | xxxx | xxxx | xxxx |
+----------+------+------+------+
| xxxxxxxx | xxxx | xxxx | xxxx |
+----------+------+------+------+
| xxxxxxxx | xxxx | xxxx | xxxx |
+----------+------+------+------+

MultiModalSparseArray inherit from scipy sparce matrix with the shape (n_samples, n_view_i * n_features_i)

"""
from abc import ABCMeta
import numpy as np
import numpy.ma as ma
import scipy.sparse as sp


class MultiModalData(metaclass=ABCMeta):

    @staticmethod
    def _first_validate_views_ind(views_ind, n_features):
        """Ensure proper format for views_ind and return number of views."""
        views_ind = np.array(views_ind)
        if np.issubdtype(views_ind.dtype, np.integer) and views_ind.ndim == 1:
            if len(views_ind) > 2 and np.any(views_ind[:-1] >= views_ind[1:]):
                raise ValueError("Values in views_ind must be sorted.")
            if views_ind[0] < 0 or views_ind[-1] > n_features:
                raise ValueError("Values in views_ind are not in a correct "
                                 + "range for the provided data.")
            view_mode_ = "slices"
            n_views = views_ind.shape[0]-1
        else:
            if views_ind.ndim == 1:
                if not views_ind.dtype == object:
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
        if np.issubdtype(views_ind.dtype, np.integer) and views_ind.ndim == 1:
            if len(views_ind) > 2 and np.any(views_ind[:-1] >= views_ind[1:]):
                raise ValueError("Values in views_ind must be sorted.")
            if views_ind[0] < 0 or views_ind[-1] > n_features:
                raise ValueError("Values in views_ind are not in a correct "
                                 + "range for the provided data.")
            self.view_mode_ = "slices"
            n_views = views_ind.shape[0]-1
        else:
            if views_ind.ndim == 1:
                if not views_ind.dtype == object:
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

    def __init__(self, data, views_ind=None):
        """Constructor of Metriclearn_array"""
        shapes_int = []
        index = 0
        new_data = np.ndarray([])
        n_views = data.size
        thekeys = None
        view_mode = 'slices'

        if (sp.issparse(data)) and data.ndim > 1:
            if  views_ind is not None:
                try:
                    views_ind = np.asarray(views_ind)
                except :
                    raise TypeError("n_views should be list or nparray")
            elif views_ind is None:
                if data.shape[1] > 1:
                    views_ind = np.array([0, data.shape[1]//2, data.shape[1]])
                else:
                    views_ind = np.array([0, data.shape[1]])

            new_data = data
        views_ind, n_views, view_mode = self._first_validate_views_ind(views_ind,
                                                                      data.shape[1])
        if views_ind.ndim == 1 and view_mode.startswith("slicing"):
            shapes_int = [in2 - in1 for in1, in2 in zip(views_ind, views_ind[1:])]

        if data.shape[0] < 1 or data.shape[1] < 1:
            raise ValueError("input data shouldbe not empty")
        self.view_mode_ = view_mode
        self.views_ind = views_ind
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

        views_ind : list of views' indice  (may be None)

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



class MultiModalArray(np.ndarray, MultiModalData):
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

    views_ind : list of views' indice  (may be None)

    n_views : int number of views

    shapes_int: list of int numbers of feature for each views


    :Example:

    >>> from multimodal.datasets.base import load_dict
    >>> from multimodal.tests.datasets.get_dataset_path import get_dataset_path
    >>> from multimodal.datasets.data_sample import DataSample
    >>> file = 'input_x_dic.pkl'
    >>> data = load_dict(get_dataset_path(file))
    >>> print(data.__class__)
    <class 'dict'>
    >>> multiviews = MultiModalArray(data)
    >>> multiviews.shape
    (120, 240)
    >>> multiviews.shapes_int
    [120, 120]
    >>> multiviews.n_views
    2


    """
    def __new__(cls, data, views_ind=None):
        """Constructor of MultiModalArray_array"""
        shapes_int = []
        index = 0
        new_data = np.ndarray([])
        n_views = 1
        thekeys = None
        view_mode = 'slices'
        if isinstance(data, dict) and not isinstance(data, sp.dok_matrix):
            n_views = len(data)
            views_ind = [0]
            for key, dat_values in data.items():
                dat_values = np.asarray(dat_values)
                if dat_values.ndim < 2:
                    dat_values = dat_values.reshape(1, dat_values.shape[0])
                new_data = cls._populate_new_data(index, dat_values, new_data)
                shapes_int.append(dat_values.shape[1])
                views_ind.append(dat_values.shape[1] + views_ind[index])
                index += 1
            thekeys = data.keys()
            if new_data.ndim > 1 and (new_data.shape == (1, 1) or new_data.shape == ()):
                raise ValueError('Reshape your data')
        elif isinstance(data, np.ndarray) and views_ind is None and data.ndim == 1:
            try:
                dat0 = np.array(data[0])
            except Exception:  # pragma: no cover
                raise TypeError("input format is not supported")

            if dat0.ndim < 2:
                data = data[np.newaxis, ...]
                if data.shape[1] > 1:
                    views_ind = np.array([0, data.shape[1]//2, data.shape[1]])
                else:
                    views_ind = np.array([0, data.shape[1]])
                new_data = data
            else:
                new_data, shapes_int, views_ind = cls._for_data(cls, data)
            n_views = data.shape[0]
        elif (isinstance(data, np.ndarray) ) and data.ndim > 1:
            try:
                data = np.asarray(data)
            except:  # pragma: no cover
                raise TypeError("input format is not supported")

            if  views_ind is not None:
                try:
                    views_ind = np.asarray(views_ind)
                except :  # pragma: no cover
                    raise TypeError("n_views should be list or nparray")
            elif views_ind is None:
                if data.shape[1] > 1:
                    views_ind = np.array([0, data.shape[1]//2, data.shape[1]])
                else:
                    views_ind = np.array([0, data.shape[1]])
            new_data = data
        else:
            try:
                new_data = np.asarray(data)
                if views_ind is None:
                    if new_data.shape[1] > 1:
                        views_ind = np.array([0, new_data.shape[1] // 2, new_data.shape[1]])
                    else:
                        views_ind = np.array([0, new_data.shape[1]])
                elif not isinstance(views_ind, np.ndarray):
                    try:
                       views_ind = np.asarray(views_ind)
                    except Exception:  # pragma: no cover
                        raise TypeError("format of views_ind is not list or nd array")
            except  Exception as e:
                raise ValueError('Reshape your data')
            if new_data.ndim < 2 :
                raise ValueError('Reshape your data')
            if  new_data.ndim > 1 and (new_data.shape == (1, 1) or new_data.shape == ()):
                raise ValueError('Reshape your data')

        views_ind, n_views, view_mode = cls._first_validate_views_ind(views_ind,
                                                                      new_data.shape[1])
        if views_ind.ndim == 1 and view_mode.startswith("slices"):
            shapes_int = [in2 - in1 for in1, in2 in zip(views_ind, views_ind[1:])]

        if hasattr(new_data, "mask"):  # pragma: no cover
            obj = ma.masked_array(new_data.data, new_data.mask).view(cls)
        elif hasattr(new_data, "data") and \
                hasattr(new_data, "shape") and len(new_data.shape) > 0:
                obj = np.asarray(new_data.data).view(cls)
        else:
            obj = np.recarray.__new__(cls, shape=(0, 0), dtype=float)
        obj.view_mode_ = view_mode
        obj.views_ind = views_ind
        obj.shapes_int = shapes_int
        obj.n_views = n_views
        return obj

    @staticmethod
    def _for_data(cls, data):
        n_views = data.shape[0]
        index = 0
        views_ind = np.empty(n_views + 1, dtype=int)
        views_ind[0] = 0
        shapes_int = []
        new_data = np.ndarray([])
        for dat_values in data:
            try:
                dat_values = np.array(dat_values)
            except Exception:  # pragma: no cover
                raise TypeError("input format is not supported")
            new_data = cls._populate_new_data(index, dat_values, new_data)
            views_ind[index + 1] = dat_values.shape[1] + views_ind[index]
            shapes_int.append(dat_values.shape[1])
            index += 1
        return new_data, shapes_int, views_ind

    @staticmethod
    def _populate_new_data(index, dat_values, new_data):  # pragma: no cover
        """Work in progress : including missing data"""
        if index == 0:
            if isinstance(dat_values, ma.MaskedArray)  or \
                  isinstance(dat_values, np.ndarray) or sp.issparse(dat_values):
                new_data = dat_values
            else:
                new_data = dat_values.view(np.ndarray)
        else:
            if isinstance(dat_values, np.ndarray):
                new_data = np.hstack((new_data, dat_values))
            elif isinstance(dat_values, ma.MaskedArray):
                new_data = ma.hstack((new_data, dat_values))
            elif sp.issparse(dat_values):
                new_data = sp.hstack((new_data, dat_values))
            else:
                new_data = np.hstack((new_data,  dat_values.view(np.ndarray) ) )
        return new_data

    def __array_finalize__(self, obj):
        if obj is None: return
        self.shapes_int = getattr(obj, 'shapes_int', None)
        self.n_views = getattr(obj, 'n_views', None)
        self.views_ind = getattr(obj, 'views_ind', None)
        self.view_mode_ = getattr(obj, 'view_mode_', None)

    def __reduce__(self):
        # Get the parent's __reduce__ tuple
        pickled_state = super(MultiModalArray, self).__reduce__()
        # Create our own tuple to pass to __setstate__
        new_state = pickled_state[2] + (self.__dict__,)
        # Return a tuple that replaces the parent's __setstate__ tuple with our own
        return (pickled_state[0], pickled_state[1], new_state)

    def __setstate__(self, state):
        self.__dict__.update(state[-1])
        super(MultiModalArray, self).__setstate__(state[0:-1])

    def get_col(self, view, col):
        start = np.sum(np.asarray(self.shapes_int[0: view]))
        return self[start+col, :]

    def get_view(self, view):
        start = int(np.sum(np.asarray(self.shapes_int[0: view])))
        stop = int(start + self.shapes_int[view])
        return self[:, start:stop]



    def set_view(self, view, data):
        start = int(np.sum(np.asarray(self.shapes_int[0: view])))
        stop = int(start + self.shapes_int[view])
        if stop-start == data.shape[1] and data.shape[0] == self.data.shape[0]:
             self[:, start:stop] = data
        else:
            raise ValueError(
                "shape of data does not match (%d, %d)" %(stop-start ,self.data.shape[0]))

    def get_row(self, view, row):
        start = int(np.sum(np.asarray(self.shapes_int[0: view])))
        stop = int(np.sum(np.asarray(self.shapes_int[0: view+1])))
        return self[row, start:stop]

    def _todict(self):
        dico = {}
        for view in range(self.n_views):
            dico[view] = self.get_view(view)
        return dico


class DataSample(dict):
    """
    A DataSample instance


    :Example:

    >>> from multimodal.datasets.base import load_dict
    >>> from multimodal.tests.datasets.get_dataset_path import get_dataset_path
    >>> from multimodal.datasets.data_sample import DataSample
    >>> file = 'input_x_dic.pkl'
    >>> data = load_dict(get_dataset_path(file))
    >>> print(data.__class__)
    <class 'dict'>
    >>> s = DataSample(data)
    >>> type(s.data)
    <class 'multimodal.datasets.data_sample.MultiModalArray'>


    - Input:

    Parameters
    ----------
    data : dict
    kwargs : others arguments

    Attributes
    ----------

    data   : { array like}  MultiModalArray
    """

    def __init__(self, data=None, **kwargs):

        # The dictionary that contains the sample
        super(DataSample, self).__init__(kwargs)
        self._data = None # Metriclearn_arrayMultiModalArray(np.zeros((0,0)))
        if data is not None:
            self._data = MultiModalArray(data)


    @property
    def data(self):
        """MultiModalArray"""

        return self._data

    @data.setter
    def data(self, data):
        if isinstance(data, (MultiModalArray, np.ndarray, ma.MaskedArray, np.generic)) or sp.issparse(data):
            self._data = data
        else:
            raise TypeError("sample should be a MultiModalArray or numpy array.")




