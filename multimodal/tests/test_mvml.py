# -*- coding: utf-8 -*-

import pickle
import unittest

import numpy as np
from sklearn.exceptions import NotFittedError

from multimodal.datasets.data_sample import MultiModalArray
from multimodal.kernels.mvml import MVML
from multimodal.tests.datasets.get_dataset_path import get_dataset_path


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
        w_expected = np.array([0.73849765, 0.52974952])
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
        with self.assertRaises(TypeError):
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
        with self.assertRaises(TypeError):
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
        with self.assertRaises(TypeError):
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
        self.assertEqual(pred.shape, (80,1))

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
        self.assertEqual(pred.shape, (80,1))

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
       self.assertEqual(pred.shape, (80,1))

    def testPredictMVML(self):
       mvml = MVML(lmbda=0.1, eta=1,
                   nystrom_param=1.0, learn_A=4)
       mvml.fit(self.kernel_dict, y=self.y)
       pred = mvml.predict(self.test_kernel_dict)
       self.assertEqual(pred.shape, (80,1))
       # print(pred.shape)



if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    # MVMLTest.testFitMVMLMetric_PredictA2()