# -*- coding: utf-8 -*-

"""This module contains the DataSample class and Metriclearn_array class
The DataSample class encapsulates a sample 's components
nbL and nbEx numbers,
Metriclearn_arra class inherit from numpy ndarray and contains a 2d data ndarray
with the shape (n_samples, n_view_i * n_features_i)

0        1    2    3
======== ==== ==== ====
xxxxxxxx xxxx xxxx xxxx
xxxxxxxx xxxx xxxx xxxx
xxxxxxxx xxxx xxxx xxxx
xxxxxxxx xxxx xxxx xxxx
xxxxxxxx xxxx xxxx xxxx
xxxxxxxx xxxx xxxx xxxx
xxxxxxxx xxxx xxxx xxxx
xxxxxxxx xxxx xxxx xxxx
xxxxxxxx xxxx xxxx xxxx
======== ==== ==== ====

the number nbL and nbEx and , the fourth dictionaries for sample,
prefix, suffix and factor where they are computed
"""
import numpy as np
import numpy.ma as ma


class Metriclearn_array(ma.MaskedArray, np.ndarray):
    """
    Metriclearn_array inherit from numpy ndarray


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

    >>> from metriclearning.datasets.base import load_dict
    >>> from metriclearning.tests.datasets.get_dataset_path import get_dataset_path
    >>> from metriclearning.datasets.data_sample import DataSample
    >>> file = 'input_x_dic.pkl'
    >>> data = load_dict(get_dataset_path(file))
    >>> print(data.__class__)
    <class 'dict'>
    >>> metric = Metriclearn_array(data)
    >>> metric.shape
    (120, 240)
    >>> metric.keys
    dict_keys([0, 1])
    >>> metric.shapes_int
    [120, 120]
    >>> metric.n_views
    2


    """
    def __new__(cls, data, view_ind=None):
        """Constructor of Metriclearn_array"""
        shapes_int = []
        index = 0
        new_data = np.ndarray([])
        n_views = len(data)
        thekeys = None
        view_ind_self =  None
        if isinstance(data, dict):
            n_views = len(data)
            for key, dat_values in data.items():
                new_data = cls._populate_new_data(index, dat_values, new_data)
                shapes_int.append(dat_values.shape[1])
                index += 1
            thekeys = data.keys()
        if isinstance(data, np.ndarray) and view_ind is None and data.ndim == 1:
            n_views = data.shape[0]
            for dat_values in data:
                shapes_int.append(dat_values.shape[1])
                new_data = cls._populate_new_data(index, dat_values, new_data)
                index += 1
        elif isinstance(data, np.ndarray) and data.ndim > 1:
            if  view_ind is not None:
                try:
                    view_ind = np.asarray(view_ind)
                except :
                    raise TypeError("n_views should be list or nparray")
                n_views = view_ind.shape[0] - 1
            elif view_ind is None:
                if data.shape[1] > 1:
                    view_ind = np.array([0, data.shape[1]//2, data.shape[1]])
                else:
                    view_ind = np.array([0, data.shape[1]])
                view_ind, n_views = cls._validate_views_ind(view_ind,
                                                            data.shape[1])
            shapes_int = [  in2-in1  for in1, in2 in  zip(view_ind, view_ind[1: ])]
            new_data = data
            view_ind_self = view_ind

        # obj =   ma.MaskedArray.__new(new_data)   # new_data.view()  a.MaskedArray(new_data, mask=new_data.mask).view(cls)
        # bj = super(Metriclearn_array, cls).__new__(cls, new_data.data, new_data.mask)
        if hasattr(new_data, "mask"):
            obj = ma.masked_array(new_data.data, new_data.mask).view(cls)
        elif hasattr(new_data, "data") and \
                hasattr(new_data, "shape") and len(new_data.shape) > 0:
            obj = np.asarray(new_data.data).view(cls)
        else:
            obj = np.recarray.__new__(cls, shape=(), dtype=np.float)
        obj.views_ind = view_ind_self
        obj.shapes_int = shapes_int
        obj.n_views = n_views
        obj.keys = thekeys
        return obj

    @staticmethod
    def _populate_new_data(index, dat_values, new_data):
        if index == 0:
            if isinstance(dat_values, ma.MaskedArray)  or isinstance(dat_values, np.ndarray):
                new_data = dat_values
            else:
                new_data = dat_values.view(ma.MaskedArray) #  ma.masked_array(dat_values, mask=ma.nomask) dat_values.view(ma.MaskedArray) #(
                new_data.mask = ma.nomask
        else:
            if isinstance(dat_values, ma.MaskedArray) or isinstance(dat_values, np.ndarray):
                new_data = ma.hstack((new_data, dat_values))
            else:
                new_data = ma.hstack((new_data,  dat_values.view(ma.MaskedArray) ) ) #  ma.masked_array(dat_values, mask=ma.nomask
        return new_data

    def __array_finalize__(self, obj):
        if obj is None: return
        super(Metriclearn_array, self).__array_finalize__(obj)
        self.shapes_int = getattr(obj, 'shapes_int', None)
        self.n_views = getattr(obj, 'n_views', None)
        self.keys = getattr(obj, 'keys', None)
        self.views_ind_self = getattr(obj, 'views_ind_self', None)

    def get_col(self, view, col):
        start = np.sum(np.asarray(self.shapes_int[0: view]))
        return self.data[start+col, :]

    def get_view(self, view):
        start = int(np.sum(np.asarray(self.shapes_int[0: view])))
        stop = int(start + self.shapes_int[view])
        return self.data[:, start:stop]

    def set_view(self, view, data):
        start = int(np.sum(np.asarray(self.shapes_int[0: view])))
        stop = int(start + self.shapes_int[view])
        if stop-start == data.shape[0] and data.shape[1]== self.data.shape[1]:
             self.data[:, start:stop] = data
        else:
            raise ValueError(
                "shape of data does not match (%d, %d)" %stop-start %self.data.shape[1])

    def get_raw(self, view, raw):
        start = np.sum(np.asarray(self.shapes_int[0: view]))
        stop = np.sum(np.asarray(self.shapes_int[0: view+1]))
        return self.data[start:stop, raw]

    def add_view(self, v, data):
        if len(self.shape) > 0:
            if data.shape[0] == self.data.shape[0]:
                indice = self.shapes_int[v]
                np.insert(self.data, data, indice+1, axis=0)
                self.shapes_int.append(data.shape[1])
                self.n_views +=1
        else:
            raise ValueError("New view can't initialazed")
           # self.shapes_int= [data.shape[1]]
           # self.data.reshape(data.shape[0],)
           # np.insert(self.data, data, 0)
           # self.n_views = 1

    def _todict(self):
        dico = {}
        for view in range(self.n_views):
            dico[view] = self.get_view(view)
        return dico

    @staticmethod
    def _validate_views_ind(views_ind, n_features):
        """Ensure proper format for views_ind and return number of views."""
        views_ind = np.array(views_ind)
        if np.issubdtype(views_ind.dtype, np.integer) and views_ind.ndim == 1:
            if np.any(views_ind[:-1] >= views_ind[1:]):
                raise ValueError("Values in views_ind must be sorted.")
            if views_ind[0] < 0 or views_ind[-1] > n_features:
                raise ValueError("Values in views_ind are not in a correct "
                                 + "range for the provided data.")
            n_views = views_ind.shape[0]-1
        else:
            raise ValueError("The format of views_ind is not "
                                     + "supported.")

        return (views_ind, n_views)


class DataSample(dict):
    """
    A DataSample instance


    :Example:

    >>> from metriclearning.datasets.base import load_dict
    >>> from metriclearning.tests.datasets.get_dataset_path import get_dataset_path
    >>> from metriclearning.datasets.data_sample import DataSample
    >>> file = 'input_x_dic.pkl'
    >>> data = load_dict(get_dataset_path(file))
    >>> print(data.__class__)
    <class 'dict'>
    >>> s = DataSample(data)
    >>> type(s.data)
    <class 'metriclearning.datasets.data_sample.Metriclearn_array'>


    - Input:

    Parameters
    ----------
    data : dict
    kwargs : others arguments

    Attributes
    ----------

    data   : { array like}  Metriclearn_array
    """

    def __init__(self, data=None, **kwargs):


        # The dictionary that contains the sample
        super(DataSample, self).__init__(kwargs)
        self._data = None # Metriclearn_array(np.zeros((0,0)))
        if data is not None:
            self._data = Metriclearn_array(data)


    @property
    def data(self):
        """Metriclearn_array"""

        return self._data

    @data.setter
    def data(self, data):
        if isinstance(data, (Metriclearn_array, np.ndarray, ma.MaskedArray, np.generic)):
            self._data = data
        else:
            raise TypeError("sample should be a Metriclearn_array.")




