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
import pickle
import unittest

import numpy as np
import scipy as sp
from sklearn.exceptions import NotFittedError

from multimodal.datasets.data_sample import MultiModalArray
from multimodal.kernels.mvml import MVML
from multimodal.tests.datasets.get_dataset_path import get_dataset_path
from sklearn.utils.estimator_checks import check_estimator

class MVMLTest(unittest.TestCase):

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


    def testInitMVML(self):
        mvml = MVML(lmbda=0.1, eta=1, nystrom_param=0.2)
        self.assertEqual(mvml.nystrom_param, 0.2)
        self.assertEqual(mvml.learn_A, 1)
        self.assertEqual(mvml.learn_w, 0)
        self.assertEqual(mvml.n_loops, 6)
        self.assertEqual(mvml.lmbda, 0.1)
        self.assertEqual(mvml.eta, 1)

    def testFitMVMLDict(self):
        #######################################################
        # task with dict and not precomputed
        #######################################################
        mvml = MVML(lmbda=0.1, eta=1,
                    kernel=['rbf'], kernel_params=[{'gamma':50}],
                    nystrom_param=0.2)
        views_ind = [120, 240]
        mvml.fit(self.kernel_dict, y=self.y, views_ind=None)
        self.assertEqual(mvml.A.shape, (48, 48))
        self.assertEqual(mvml.g.shape,(48, 1))
        w_expected = np.array([[0.5],[0.5]])
        np.testing.assert_almost_equal(mvml.w, w_expected, 8)

    def testFitMVMLRegression(self):
        #######################################################
        # task with dict and not precomputed
        #######################################################
        y = self.y
        y += np.random.uniform(0,1, size=y.shape)
        mvml = MVML(lmbda=0.1, eta=1,
                    kernel=['rbf'], kernel_params=[{'gamma':50}],
                    nystrom_param=0.2)
        views_ind = [120, 240]
        mvml.fit(self.kernel_dict, y=y, views_ind=None)
        self.assertEqual(mvml.A.shape, (48, 48))
        self.assertEqual(mvml.g.shape,(48, 1))
        w_expected = np.array([[0.5],[0.5]])
        np.testing.assert_almost_equal(mvml.w, w_expected, 8)

    def testFitMVMLPrecision(self):
        #######################################################
        # task with dict and not precomputed
        #######################################################
        mvml = MVML(lmbda=0.1, eta=1,
                    kernel=['rbf'], kernel_params=[{'gamma':50}],
                    nystrom_param=0.2, precision=1E-0)
        views_ind = [120, 240]
        mvml.fit(self.kernel_dict, y=self.y, views_ind=None)
        self.assertEqual(mvml.A.shape, (48, 48))
        self.assertEqual(mvml.g.shape,(48, 1))
        w_expected = np.array([[0.5],[0.5]])
        np.testing.assert_almost_equal(mvml.w, w_expected, 8)

    def testFitMVMLDictNLoop(self):
        #######################################################
        # task with dict and not precomputed
        #######################################################
        mvml = MVML(lmbda=0.1, eta=1,
                    kernel=['rbf'], kernel_params=[{'gamma':50}],
                    nystrom_param=0.2, n_loops=2, learn_w=1)
        views_ind = [120, 240]
        mvml.fit(self.kernel_dict, y=self.y, views_ind=None)
        self.assertEqual(mvml.A.shape, (48, 48))
        self.assertEqual(mvml.g.shape,(48, 1))
        w_expected = np.array([0.655, 0.65 ])
        np.testing.assert_almost_equal(mvml.w, w_expected, 3)

    def testFitMVMLMetric(self):
        #######################################################
        # task with Metric array
        #######################################################
        # mvml = MVML.fit(self.kernel_dict, self.y)
        w_expected = np.array([[0.5], [0.5]])
        x_metricl = MultiModalArray(self.kernel_dict)
        mvml2 = MVML(lmbda=0.1, eta=1, nystrom_param=1.0)
        mvml2.fit(x_metricl, y=self.y, views_ind=None)
        self.assertEqual(mvml2.A.shape, (240, 240))
        self.assertEqual(mvml2.g.shape,(240, 1))
        np.testing.assert_almost_equal(mvml2.w, w_expected, 8)
        with self.assertRaises(ValueError):
            mvml2.fit([1, 2, 3])

    def testFitMVMLMetric_learA4(self):
        #######################################################
        # task with Metric array
        #######################################################
        # mvml = MVML.fit(self.kernel_dict, self.y)
        w_expected = np.array([[0.5], [0.5]])
        x_metricl = MultiModalArray(self.kernel_dict)
        mvml2 = MVML(lmbda=0.1, eta=1, nystrom_param=1.0, learn_A=4)
        mvml2.fit(x_metricl, y=self.y, views_ind=None)
        self.assertEqual(mvml2.A.shape, (240, 240))
        self.assertEqual(mvml2.g.shape,(240, 1))
        np.testing.assert_almost_equal(mvml2.w, w_expected, 8)
        with self.assertRaises(ValueError):
            mvml2.fit([1, 2, 3])

    def testFitMVMLMetric_learA3(self):
        #######################################################
        # task with Metric array
        #######################################################
        # mvml = MVML.fit(self.kernel_dict, self.y)
        w_expected = np.array([[0.5], [0.5]])
        x_metricl = MultiModalArray(self.kernel_dict)
        mvml2 = MVML(lmbda=0.1, eta=1, nystrom_param=1.0, learn_A=3)
        mvml2.fit(x_metricl, y=self.y, views_ind=None)
        self.assertEqual(mvml2.A.shape, (240, 240))
        self.assertEqual(mvml2.g.shape,(240, 1))
        np.testing.assert_almost_equal(mvml2.w, w_expected, 8)
        with self.assertRaises(ValueError):
            mvml2.fit([1, 2, 3])

    def testFitMVMLMetric_PredictA2(self):
        #######################################################
        # task with Metric array
        #######################################################
        w_expected = np.array([0.2,  0.1]) # [0.94836083 , 0.94175933] [ 0.7182,  0.7388]
        x_metricl = MultiModalArray(self.kernel_dict)
        mvml2 = MVML(lmbda=0.1, eta=1, nystrom_param=0.6,
                     learn_A=2, learn_w=1)
        mvml2.fit(x_metricl, y=self.y, views_ind=None)
        self.assertEqual(mvml2.A.shape, (144, 144))
        self.assertEqual(mvml2.g.shape,(144, 1))
        np.testing.assert_almost_equal(mvml2.w, w_expected, 0)
        pred = mvml2.predict(self.test_kernel_dict)
        self.assertEqual(pred.shape, (80,))

    def testFitMVMLMetric_PredictA1(self):
        #######################################################
        # task with Metric array
        #######################################################
        w_expected = np.array([1.3,  1.4]) # [0.94836083 , 0.94175933] [ 0.7182,  0.7388]
        x_metricl = MultiModalArray(self.kernel_dict)
        mvml2 = MVML(lmbda=0.1, eta=1, nystrom_param=0.6,
                     learn_A=1, learn_w=1)
        mvml2.fit(x_metricl, y=self.y, views_ind=None)
        self.assertEqual(mvml2.A.shape, (144, 144))
        self.assertEqual(mvml2.g.shape,(144, 1))
        np.testing.assert_almost_equal(mvml2.w, w_expected, 0)
        pred = mvml2.predict(self.test_kernel_dict)
        self.assertEqual(pred.shape, (80,))

    def testFitMVMLArray_2d(self):
        #######################################################
        # task with nparray 2d
        #######################################################
        w_expected = np.array([[0.5], [0.5]])
        x_metricl = MultiModalArray(self.kernel_dict)
        x_array = np.asarray(x_metricl)
        mvml3 = MVML(lmbda=0.1, eta=1, nystrom_param=1.0)
        mvml3.fit(x_array, y=self.y, views_ind=[0, 120, 240])
        self.assertEqual(mvml3.A.shape, (240, 240))
        self.assertEqual(mvml3.g.shape,(240, 1))
        np.testing.assert_almost_equal(mvml3.w, w_expected, 8)

    def testFitMVMLArray_1d(self):
        #######################################################
        # task with nparray 1d
        #######################################################
        w_expected = np.array([[0.5], [0.5]])
        n_views = len(self.kernel_dict)
        x_array_1d = np.ndarray((n_views), dtype=object)
        for v in range(n_views):
            x_array_1d[v] = self.kernel_dict[v]
        mvml4 = MVML(lmbda=0.1, eta=1, learn_A=3, nystrom_param=0.6,
                     kernel=['rbf'], kernel_params=[{'gamma':50}])
        mvml4.fit(x_array_1d, y=self.y)
        self.assertEqual(mvml4.A.shape, (144, 144))
        self.assertEqual(mvml4.g.shape,(144, 1))
        np.testing.assert_almost_equal(mvml4.w, w_expected, 8)


    def testPredictMVML_witoutFit(self):
       mvml = MVML(lmbda=0.1, eta=1,
                   kernel=['rbf'], kernel_params=[{'gamma':50}],
                   nystrom_param=0.2)
       with self.assertRaises(NotFittedError):
           mvml.predict(self.test_kernel_dict)

    def testPredictMVMLKernel(self):
       mvml = MVML(lmbda=0.1, eta=1,
                   kernel=['rbf'], kernel_params={'gamma':50},
                   nystrom_param=0.2, learn_A=4)
       mvml.fit(self.kernel_dict, y=self.y)
       pred =mvml.predict(self.test_kernel_dict)
       self.assertEqual(pred.shape, (80,))

    def testPredictMVML(self):
       mvml = MVML(lmbda=0.1, eta=1,
                   nystrom_param=1.0, learn_A=4)
       mvml.fit(self.kernel_dict, y=self.y)
       pred = mvml.predict(self.test_kernel_dict)
       self.assertEqual(pred.shape, (80,))
       # print(pred.shape)

    def test_classifier(self):
        pass
        # return check_estimator(MVML())

    def test_check_kernel(self):
        clf = MVML()
        clf.kernel = "an_unknown_kernel"
        self.assertRaises(ValueError, clf._check_kernel)

    def testFitMVMLSparesArray(self):
        #######################################################
        # task with nparray 2d
        #######################################################
        x_metricl = MultiModalArray(self.kernel_dict)
        x_array = np.asarray(x_metricl)
        x_array_sparse = sp.sparse.csr_matrix(x_array)
        mvml3 = MVML(lmbda=0.1, eta=1, nystrom_param=1.0)
        self.assertRaises(TypeError, mvml3.fit, x_array_sparse, self.y, [0, 120, 240])

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    # MVMLTest.testFitMVMLMetric_PredictA2()