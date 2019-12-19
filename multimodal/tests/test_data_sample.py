import unittest
import numpy as np

from metriclearning.datasets.base import load_dict
from metriclearning.tests.datasets.get_dataset_path import get_dataset_path
from metriclearning.datasets.data_sample import Metriclearn_array
import pickle

class UnitaryTest(unittest.TestCase):

    @classmethod
    def setUpClass(clf):
        input_x = get_dataset_path("input_x_dic.pkl")
        f = open(input_x, "rb")
        kernel_dict = pickle.load(f)
        f.close()
        test_input_x = get_dataset_path("test_kernel_input_x.pkl")
        f = open(test_input_x, "rb")
        test_kernel_dict = pickle.load(f)
        f.close()
        test_input_y = get_dataset_path("test_input_y.npy")
        input_y = get_dataset_path("input_y.npy")
        y = np.load(input_y)
        test_y = np.load(test_input_y)
        clf.y = y
        clf.kernel_dict = kernel_dict
        clf.test_kernel_dict = test_kernel_dict
        clf.test_y = test_y


    def testGet_view(self):
        a = Metriclearn_array(self.kernel_dict)
        np.testing.assert_almost_equal(a.get_view(0), self.kernel_dict[0], 8)
        np.testing.assert_almost_equal(a.get_view(1), self.kernel_dict[1], 8)

    def test_init_Metriclearn_array(self):
        a = Metriclearn_array(self.kernel_dict)
        self.assertEqual(a.shape, (120, 240))
        self.assertEqual(a.shapes_int, [120, 120])
        self.assertEqual(a.n_views, 2)
        dict_key = {0: 'a',1: 'b' }
        self.assertEqual(a.keys, dict_key.keys())

    def test_init_Array(self):
        a = Metriclearn_array(self.kernel_dict)
        array_x = a.data
        b = Metriclearn_array(a)
        np.testing.assert_equal(b.views_ind, np.array([0, 120, 240]))


