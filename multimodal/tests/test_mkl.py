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
from sklearn.metrics.pairwise import rbf_kernel
from multimodal.tests.datasets.get_dataset_path import get_dataset_path
from multimodal.kernels.lpMKL import MKL
from multimodal.datasets.data_sample import MultiModalArray
import pickle
from sklearn.exceptions import NotFittedError


class MKLTest(unittest.TestCase):

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

    def testInitMKL(self):
        mkl = MKL(lmbda=3, nystrom_param=1.0, kernel = "precomputed",
                   kernel_params = None, use_approx = True,
                   precision = 1E-4, n_loops = 50)
        self.assertEqual(mkl.nystrom_param, 1.0)
        self.assertEqual(mkl.lmbda, 3)
        self.assertEqual(mkl.n_loops, 50)
        self.assertEqual(mkl.precision, 1E-4)

    def testFitMKLDict(self):
        #######################################################
        # task with dict and not precomputed
        #######################################################
        mkl = MKL(lmbda=3, nystrom_param=1.0, kernel=['rbf'], kernel_params=[{'gamma':50}],
                   use_approx = True,
                   precision = 1E-4, n_loops = 50)
        mkl.fit(self.kernel_dict, y=self.y, views_ind=None)
        self.assertEqual(mkl.C.shape, (120,))
        np.testing.assert_almost_equal(mkl.weights, np.array([0.70629782, 0.70791482]), 8)


    def testFitMKLDictNLoop(self):
        #######################################################
        # task with dict and not precomputed
        #######################################################
        mkl = MKL(lmbda=3, nystrom_param=0.3, kernel=['rbf'], kernel_params=[{'gamma':50}],
                   use_approx=True,
                   precision = 1E-4, n_loops = 50)
        views_ind = [120, 240]
        mkl.fit(self.kernel_dict, y=self.y, views_ind=None)
        self.assertEqual(mkl.C.shape, (120,))
        np.testing.assert_almost_equal(mkl.weights, np.array([0.70629782, 0.70791482]), 8)

    def testFitMKLMetricPrecision(self):
        #######################################################
        # task with Metric array
        #######################################################
        # mvml = MVML.fit(self.kernel_dict, self.y)
        w_expected = np.array([[0.5], [0.5]])
        x_metricl = MultiModalArray(self.kernel_dict)
        mkl2 = MKL(lmbda=3, nystrom_param = 0.3, kernel=['rbf'], kernel_params=[{'gamma':50}],
                   use_approx = True,
                   precision = 1E0, n_loops = 50)
        with self.assertRaises(ValueError):
            mkl2.fit(x_metricl, y=self.y, views_ind=None)

    def testFitMKLMetricPrecision2(self):
        #######################################################
        # task with Metric array
        #######################################################
        # mvml = MVML.fit(self.kernel_dict, self.y)
        w_expected = np.array([[0.5], [0.5]])
        x_metricl = MultiModalArray(self.kernel_dict)
        mkl2 = MKL(lmbda=3, nystrom_param=0.3, kernel="precomputed",
                   use_approx = True,
                   precision = 1E-9, n_loops = 600)
        mkl2.fit(x_metricl, y=self.y, views_ind=None)

    def testPredictMVML_witoutFit(self):
       mkl = MKL(lmbda=3, nystrom_param=0.3, kernel=['rbf'], kernel_params=[{'gamma':50}],
                   use_approx = True,
                   precision = 1E-9, n_loops = 50)
       with self.assertRaises(NotFittedError):
            mkl.predict(self.test_kernel_dict)

    def testPredictMVML_witoutFit(self):
       x_metric = MultiModalArray(self.kernel_dict)
       mkl = MKL(lmbda=3, nystrom_param = 0.3, kernel=['rbf'], kernel_params=[{'gamma':50}],
                   use_approx = True,
                   precision = 1E-9, n_loops = 50)
       mkl.fit(x_metric, y=self.y, views_ind=None)
       pred =mkl.predict(self.test_kernel_dict)
       self.assertEqual(pred.shape, (80,))

