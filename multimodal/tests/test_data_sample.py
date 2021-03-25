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
import unittest
import numpy as np

from multimodal.datasets.base import load_dict
from multimodal.tests.datasets.get_dataset_path import get_dataset_path
from multimodal.datasets.data_sample import MultiModalArray
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
        a = MultiModalArray(self.kernel_dict)
        np.testing.assert_almost_equal(a.get_view(0), self.kernel_dict[0], 8)
        np.testing.assert_almost_equal(a.get_view(1), self.kernel_dict[1], 8)

    def test_init_Multimodal_array(self):
        a = MultiModalArray(self.kernel_dict)
        self.assertEqual(a.shape, (120, 240))
        self.assertEqual(a.shapes_int, [120, 120])
        self.assertEqual(a.n_views, 2)

    def test_init_Array(self):
        a = MultiModalArray(self.kernel_dict)
        array_x = a.data
        b = MultiModalArray(a)
        np.testing.assert_equal(b.views_ind, np.array([0, 120, 240]))
        view_1 = np.random.randint(1,10,10)
        view_2 = np.random.randint(1,10,11)
        data = {0 : view_1,
                1 : view_2,}
        c = MultiModalArray(data)
        np.testing.assert_array_equal(np.asarray(c[0,:]), np.concatenate((view_1, view_2)))
        view_1 = np.random.randint(1, 10, 1)
        data = {0: view_1, }
        self.assertRaises(ValueError, MultiModalArray, data)
        view_1 = np.array([0,])
        data = view_1
        d = MultiModalArray(data)
        np.testing.assert_array_equal(d.views_ind, np.array([0,1]))
        view_1 = [[0]]
        data = view_1
        self.assertRaises(ValueError, MultiModalArray, data)
        view_1 = [[0,1,2 ],[0,1,2 ]]
        data = view_1
        d = MultiModalArray(data, [0,1])
        np.testing.assert_array_equal(d.views_ind, np.array([0, 1]))
        view_1 = [[0]]
        data = view_1
        self.assertRaises(ValueError, MultiModalArray, data)

    def test_view_functions(self):
        view_1 = np.random.randint(1, 10, (5, 10))
        view_2 = np.random.randint(1, 10, (5, 10))
        view_3 = np.random.randint(1, 10, (5, 10))
        data = {0: view_1,
                1: view_2, }
        c = MultiModalArray(data)
        c.set_view(1, view_3)
        np.testing.assert_array_equal(c.get_view(1), view_3)
        view_3 = np.random.randint(1, 10, (12, 10))
        c = MultiModalArray(data)
        self.assertRaises(ValueError, c.set_view, 1, view_3)
        np.testing.assert_array_equal(c.get_row(0, 2), view_1[2, :])



if __name__ == '__main__':
    unittest.main()