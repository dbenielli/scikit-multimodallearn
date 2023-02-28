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
#
#
#
#
#
"""Testing for the mumbo module."""


import pickle
import numpy as np
import unittest
from scipy.sparse import csc_matrix, csr_matrix, coo_matrix, dok_matrix
from scipy.sparse import lil_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.utils.estimator_checks import check_estimator
from multimodal.boosting.combo import MuComboClassifier
from multimodal.tests.data.get_dataset_path import get_dataset_path
from multimodal.datasets.data_sample import MultiModalArray

class NoSampleWeightLasso(Lasso):

    def fit(self, X, y, check_input=True):
        return Lasso.fit(self, X, y, sample_weight=None, check_input=True)


class TestMuComboClassifier(unittest.TestCase):

    @classmethod
    def setUpClass(clf):
        # Load the iris dataset
        iris = datasets.load_iris()
        iris.views_ind = np.array([0, 2, 4])
        clf.iris = iris


    def test_init(self):
        clf = MuComboClassifier()
        self.assertEqual(clf.random_state, None)
        self.assertEqual(clf.n_estimators, 50)
        n_estimators = 3
        clf = MuComboClassifier(n_estimators=n_estimators)
        #self.assertEqual(clf.view_mode_)

    def test_init_var(self):
        n_classes = 3
        n_views = 3
        y = np.array([0, 2, 1, 2])
        expected_cost = np.array(
            [[[-2, 1, 0.5], [1, 1, -1], [1, -2, 0.5], [1, 1, -1]],
             [[-2, 1, 0.5], [1, 1, -1], [1, -2, 0.5], [1, 1, -1]],
             [[-2, 1, 0.5], [1, 1, -1], [1, -2,0.5], [1, 1, -1]]],
            dtype=np.float64)
        # expected_cost_glob = np.array(
        #     [[-2, 1, 1], [1, 1, -2], [1, -2, 1], [1, 1, -2]], dtype=np.float64)
        expected_label_score = np.zeros((n_views, y.shape[0], n_classes))
        expected_label_score_glob = np.zeros((y.shape[0], n_classes))
        expected_predicted_classes_shape = ((n_views, y.shape[0]))
        expected_n_yi_s = np.array([1, 1, 2])
        expected_beta_class = np.ones((n_views, n_classes)) / n_classes
        clf = MuComboClassifier()
        clf.n_classes_ = n_classes
        (cost,  label_score, label_score_glob, predicted_classes, score_function, beta_class, n_yi_s) \
            = clf._init_var(n_views, y)
        np.testing.assert_equal(cost, expected_cost)
        # np.testing.assert_equal(cost_glob, expected_cost_glob)
        np.testing.assert_equal(label_score, expected_label_score)
        np.testing.assert_equal(label_score_glob, expected_label_score_glob)
        np.testing.assert_equal(predicted_classes.shape, expected_predicted_classes_shape)
        np.testing.assert_equal(n_yi_s, expected_n_yi_s)
        np.testing.assert_equal(beta_class, expected_beta_class)
        np.testing.assert_equal(score_function, np.zeros((n_views, 4, n_classes)))

    def test_compute_dist(self):
        cost = np.array(
            [[[-2, 1, 1], [-1, -1, -2], [1, -2, 1], [1, 1, -2]],
             [[-1, 2, 2], [2, 2, -1], [-2, 4, -2], [2, 2, -4]],
             [[1, 4, -4], [-1, 3, -1], [-2, 2, 4], [4, 4, -4]]],
            dtype=np.float64)
        y = np.array([0, 2, 1, 2])
        expected_dist = np.array(
            [[0.25, 0.25, 0.25, 0.25], [0.5, 0.5, -2., 2.], [-0.5, 0.5, -1., 2.]])

        clf = MuComboClassifier()
        dist = clf._compute_dist(cost, y)

        np.testing.assert_equal(dist, expected_dist)

        # The computation of the distribution only uses the costs when predicting
        # the right classes, so the following cost matrix should give the same
        # result as the previous.
        cost = np.array(
            [[[-2, 0, 0], [0, 0, -2], [0, -2, 0], [0, 0, -2]],
             [[-1, 0, 0], [0, 0, -1], [0, 4, 0], [0, 0, -4]],
             [[1, 0, 0], [0, 0, -1], [0, 2, 0], [0, 0, -4]]],
            dtype=np.float64)

        dist = clf._compute_dist(cost, y)

        np.testing.assert_equal(dist, expected_dist)

        expected_cost = np.array(
            [[[-2, 1, 0.5], [1, 1, -1], [1, -2, 0.5], [1, 1, -1]],
             [[-2, 1, 0.5], [1, 1, -1], [1, -2, 0.5], [1, 1, -1]],
             [[-2, 1, 0.5], [1, 1, -1], [1, -2,0.5], [1, 1, -1]]],
            dtype=np.float64)
        dist = clf._compute_dist(cost, y)
        expected_dist = np.array([[ 0.25,0.25,0.25,0.25],
                                  [ 0.5,0.5, -2.,2. ], [-0.5 , 0.5 ,-1.,  2. ]])
        np.testing.assert_equal(dist, expected_dist)


    # def test_compute_coop_coef(self):
    #     y = np.array([0, 1, 2, 0])
    #     predicted_classes = np.array([[0, 0, 1, 1], [0, 1, 0, 2], [2, 2, 0, 0]])
    #     expected_coop_coef = np.array([[1, 0, 1, 0], [1, 1, 1, 0], [0, 0, 1, 1]],
    #                                   dtype=np.float64)
    #
    #     clf = MuComboClassifier()
    #     coop_coef = clf._compute_coop_coef(predicted_classes, y)
    #
    #     assert_array_equal(coop_coef, expected_coop_coef)


    def test_compute_edges(self):
        cost = np.array(
            [[[-2, 1, 1], [-1, -1, -2], [1, -2, 1], [1, 1, -2]],
             [[-2, 2, 2], [2, 2, -4], [-2, -4, -2], [2, 2, -4]],
             [[1, 4, -4], [-1, 3, -1], [-2, 4, 4], [4, 4, -1]]],
            dtype=np.float64)
        predicted_classes = np.array([[0, 2, 1, 1], [0, 1, 0, 2], [2, 2, 0, 1]])
        y = np.array([0, 2, 1, 2])
        expected_edges = np.array([1.25, 0.75, 0.25])

        clf = MuComboClassifier()
        edges = clf._compute_edges(cost, predicted_classes, y)

        np.testing.assert_equal(edges, expected_edges)


    def test_compute_alphas(self):
        decimal = 12
        expected_alpha = 0.5
        edge = (np.e-1.) / (np.e+1.)

        clf = MuComboClassifier()
        alpha = clf._compute_alphas(edge)
        self.assertAlmostEqual(alpha, expected_alpha, decimal)

        expected_alphas = np.array([0.5, 1., 2.])
        tmp = np.array([np.e, np.e**2, np.e**4])
        edges = (tmp-1.) / (tmp+1.)

        alphas = clf._compute_alphas(edges)

        np.testing.assert_almost_equal(alphas, expected_alphas, decimal)


    def test_prepare_beta_solver(self):
        clf = MuComboClassifier()
        clf.n_views_ = 3
        clf.n_classes_ = 3
        A, b, G, h, l = clf._prepare_beta_solver()
        a_n = np.array(A)
        A_expected = np.array([[ 1 , 1 , 1 , 0 , 0,  0,  0 , 0, 0],
                               [ 0 , 0 , 0 , 1,  1,  1,  0 , 0, 0],
                               [ 0 , 0 , 0 , 0 , 0,  0,  1,  1,1 ]])
        np.testing.assert_equal(a_n , A_expected)
        b_expected = np.array([[ 1.00e+00],[ 1.00e+00],[ 1.00e+00]])
        #                      [ 1.00e+00],[ 1.00e+00],[ 1.00e+00],
        #                      [ 1.00e+00],[ 1.00e+00],[ 1.00e+00]])
        b_n = np.array(b)
        np.testing.assert_equal(b_n, b_expected)
        G_n = np.array(G)
        G_expected = np.array([[1.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00],
                               [0.00e+00, 1.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00],
                               [0.00e+00, 0.00e+00, 1.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00],
                               [0.00e+00, 0.00e+00, 0.00e+00, 1.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00],
                               [0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 1.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00],
                               [0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 1.00e+00, 0.00e+00, 0.00e+00, 0.00e+00],
                               [0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 1.00e+00, 0.00e+00, 0.00e+00],
                               [0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 1.00e+00, 0.00e+00],
                               [0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 1.00e+00],
                               [-1.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,  0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00],
                               [0.00e+00, -1.00e+00,  0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00],
                               [0.00e+00, 0.00e+00, -1.00e+00,  0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00],
                               [0.00e+00, 0.00e+00, 0.00e+00, -1.00e+00,  0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00],
                               [0.00e+00, 0.00e+00, 0.00e+00,  0.00e+00, -1.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00],
                               [0.00e+00, 0.00e+00, 0.00e+00,  0.00e+00,  0.00e+00,-1.00e+00, 0.00e+00, 0.00e+00, 0.00e+00],
                               [0.00e+00, 0.00e+00, 0.00e+00,  0.00e+00,  0.00e+00, 0.00e+00, -1.00e+00, 0.00e+00, 0.00e+00],
                               [0.00e+00, 0.00e+00, 0.00e+00,  0.00e+00,  0.00e+00, 0.00e+00, 0.00e+00, -1.00e+00, 0.00e+00],
                               [0.00e+00, 0.00e+00, 0.00e+00,  0.00e+00,  0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,-1.00e+00]])
        np.testing.assert_equal(G_n, G_expected)
        h_n = np.array(h)
        h_expected = np.array([[ 1.00e+00],[ 1.00e+00],[ 1.00e+00],[ 1.00e+00],[ 1.00e+00],[ 1.00e+00],[ 1.00e+00],[ 1.00e+00],[ 1.00e+00],
                               [ 0.00e+00],[ 0.00e+00],[ 0.00e+00],[ 0.00e+00],[ 0.00e+00],[ 0.00e+00],[ 0.00e+00],[ 0.00e+00],[ 0.00e+00]])
        np.testing.assert_equal(h_n, h_expected)

        self.assertEqual(l, {'l': 18})

    def test_solver_cp_forbeta(self):
            clf = MuComboClassifier()
            clf.n_views_ = 3
            clf.n_classes_ = 3
            clf.n_yi_ = np.array([1, 1, 2])
            A, b, G, h, l = clf._prepare_beta_solver()
            y_i = np.array([0, 1, 2, 0])
            predicted_classes = np.array([[0, 0, 1, 1], [0, 1, 0, 2], [2, 2, 0, 0]])

            indicat , indicat_yi, delta = clf._indicatrice(predicted_classes, y_i)
            indicate_vue = np.block(np.split(indicat, 3, axis=0)).squeeze()
            indicate_vue_yi = np.block(np.split(indicat_yi, 3, axis=0)).squeeze()
            alphas = np.array([0.5, 1., 2.])
            cost_Tminus1 = 10 * np.array(
                [[[2, 1, 0.5], [1, 1, 1], [1, 2, 0.5], [1, 1, 1]],
                 [[2, 1, 0.5], [1, 1, 1], [1, 2, 0.5], [1, 1, 1]],
                 [[2, 1, 0.5], [1, 1, 1], [1, 2, 0.5], [1, 1, 1]]],
                dtype=np.float64)
            cost_Tminus1_vue = np.block(np.split(cost_Tminus1, 3, axis=0)).squeeze()
            delta_vue = np.block(np.split(delta, 3, axis=0)).squeeze()
            solver = np.array(clf._solver_cp_forbeta(alphas, indicate_vue, indicate_vue_yi, delta_vue,
                                         cost_Tminus1_vue, A, b, G, h, l))
            self.assertEqual(solver.shape, (9,1))
            s_r = np.sum(solver.reshape(3,3), axis=1)
            np.testing.assert_almost_equal(s_r, np.ones(3, dtype=float), 9)

    def test_solver_compute_betas(self):
        clf = MuComboClassifier()
        clf.n_views_ = 3
        clf.n_classes_ = 3
        clf.n_yi_ = np.array([1, 1, 2])
        cost_Tminus1 =np.array([[[-7.45744585e+01,  3.67879439e-01,  7.42065790e+01],
          [ 4.78511743e-06,  3.87742081e-02, -3.87789932e-02],
          [ 2.47875218e-03, -2.48182428e-03,  3.07210618e-06],
          [ 1.35335283e-01,  6.73794700e-03, -1.42073230e-01]],

         [[-2.02255359e-01,  1.83156389e-02,  1.83939720e-01],
          [ 7.38905610e+00,  4.97870684e-02, -7.43884317e+00],
          [ 3.67879441e-01, -4.06240749e+00,  3.69452805e+00],
          [ 3.67879441e-01,  8.10308393e+03, -8.10345181e+03]],

         [[-2.48182452e-03,  2.47875218e-03,  3.07234660e-06],
          [ 5.45938775e+01,  4.03397223e+02, -4.57991101e+02],
          [ 1.48401545e+02, -1.48426438e+02,  2.48935342e-02],
          [ 1.09663316e+03,  2.71828184e+00, -1.09935144e+03]]])
        score_function_Tminus1 =10 *np.ones((3,4,3), dtype =float)
        alphas = np.array([0.5, 1., 2.])
        predicted_classes = np.array([[0, 0, 1, 1], [0, 1, 0, 2], [2, 2, 0, 0]])
        y = np.array([0, 1, 2, 0])
        betas = clf._compute_betas(alphas, y, score_function_Tminus1, predicted_classes)
        self.assertEqual(betas.shape, (3,3))
        np.testing.assert_almost_equal(np.sum(betas, axis =1), np.ones(3, dtype=float), 9)
        self.assertTrue(np.all(betas <= 1) )
        self.assertTrue(np.all(betas >= 0) )


    def test_indicatrice(self):
        clf = MuComboClassifier()
        clf.n_views_ = 3
        clf.n_classes_ = 3
        y_i = np.array([0, 1, 2, 0])
        predicted_classes = np.array([[0, 0, 1, 1], [0, 1, 0, 2], [2, 2, 0, 0]])
        indic , indic_yi, delta = clf._indicatrice(predicted_classes, y_i)
        expected_indi = np.array( [[[0 ,0, 0], [1 ,0 ,0],[0 ,1, 0],[0 ,1 ,0]],
                                   [[0 , 0 , 0], [0,  0 ,0],  [1 ,0 ,0], [0, 0 ,1]],
                                   [[0, 0 ,1], [0 ,0 ,1],[1, 0, 0], [0, 0, 0]]])
        expected_indi_yi = np.array([[[1, 0 ,0], [0, 1, 0],  [0 ,0, 1], [1 ,0 ,0]],
                                  [[1, 0 ,0], [0 ,1 ,0], [0, 0, 1], [1, 0, 0]],
                                  [[1, 0, 0], [0 ,1, 0],  [0 ,0 ,1],  [1 ,0 ,0]]])
        np.testing.assert_equal(indic , expected_indi)
        np.testing.assert_equal(indic_yi , expected_indi_yi)

    def test_compute_cost(self):
        decimal = 12
        label_score = np.array(
            [[[-1, -2, 4], [-8, 1, 4], [2, 8, -4], [2, -1, 4]],
             [[2, -2, 1], [4, -1, 2], [1, 2, 4], [-2, 8, -1]],
             [[8, 2, -4], [2, 4, -2], [4, 1, -2], [8, 2, 1]]],
            dtype=np.float64)
        predicted_classes = np.array([[0, 2, 1, 1], [0, 1, 0, 0], [2, 2, 0, 1]])
        y = np.array([0, 2, 1, 2])
        alphas = np.array([0.25, 0.5, 2.])
        # expected_label_score = np.array(
        #      [[[-0.75, -2, 4], [-8, 1, 4.25], [2, 8.25, -4], [2, -0.75, 4]],
        #      [[2.5, -2, 1], [4, -0.5, 2], [1.5, 2, 4], [-1.5, 8, -1]],
        #       [[8, 2, -2.], [2, 4, 0.], [6., 1, -2], [8, 4., 1]]],
        #      dtype=np.float64)
        cost_Tminus1 = np.array(
                [[[-2, 1, 0.5], [1, 1, -1], [1, -2, 0.5], [1, 1, -1]],
                 [[-2, 1, 0.5], [1, 1, -1], [1, -2, 0.5], [1, 1, -1]],
                 [[-2, 1, 0.5], [1, 1, -1], [1, -2, 0.5], [1, 1, -1]]],
                dtype=np.float64)
        score_function_Tminus1 =10 *np.ones((3,4,3), dtype=float)
        clf = MuComboClassifier()
        clf.n_views_ = 3
        clf.n_classes_ = 3
        clf.n_yi_ = np.array([1, 1, 2])
        betas = clf._compute_betas(alphas, y, score_function_Tminus1, predicted_classes)
        cost, label_score, score_function_dif = clf._compute_cost(label_score, predicted_classes, y, alphas,
                                              betas, use_coop_coef=True)
        cost_expected = np.array([[[-6.58117293e+01,  3.24652469e-01,  6.54870769e+01],
                                   [ 5.42224839e-06,  4.39369338e-02, -4.39423560e-02],
                                   [ 2.47875216e-03, -2.48182426e-03,  3.07210615e-06],
                                   [ 1.35335283e-01,  6.73794705e-03, -1.42073230e-01]],

                                  [[-2.02255355e-01,  1.83156385e-02,  1.83939717e-01],
                                   [ 7.38905610e+00,  4.97870688e-02, -7.43884317e+00],
                                   [ 3.67879448e-01, -4.06240750e+00,  3.69452805e+00],
                                   [ 3.67879448e-01,  8.10308393e+03, -8.10345181e+03]],

                                  [[-2.48725300e-03,  2.47875218e-03,  8.50082247e-06],
                                   [ 1.97311865e+01,  1.45794844e+02, -1.65526031e+02],
                                   [ 3.28220396e+01, -3.28469331e+01,  2.48935342e-02],
                                   [ 1.09663316e+03,  4.44198002e+00, -1.10107514e+03]]])
        np.testing.assert_almost_equal(cost, cost_expected, 4)
        expected_label_score = np.array([[[-0.875,      -2.,          4.,        ],
                                       [-8.,          1.,          4.125     ],
                                       [ 2.,          8.00000001, -4.        ],
                                       [ 2.,         -0.99999999,  4.        ]],

                                      [[ 2.00000002, -2.,          1.        ],
                                       [ 4.,         -0.99999999,  2.        ],
                                       [ 1.00000002,  2.,          4.        ],
                                       [-1.99999998,  8.,         -1.        ]],

                                      [[ 8.,          2.,         -2.98220046],
                                       [ 2.,         4.,         -0.98220046],
                                       [ 4.49110023,  1.,         -2.        ],
                                       [ 8.,          2.49110023,  1.        ]]])
        np.testing.assert_almost_equal(label_score, expected_label_score,6)


    #
    #     label_score = np.array(
    #         [[[-1, -2, 4], [-8, 1, 4], [2, 8, -4], [2, -1, 4]],
    #          [[2, -2, 1], [4, -1, 2], [1, 2, 4], [-2, 8, -1]],
    #          [[8, 2, -4], [2, 4, -2], [4, 1, -2], [8, 2, 1]]],
    #         dtype=np.float64)
    #     expected_label_score = np.array(
    #         [[[-0.75, -2, 4], [-8, 1, 4.25], [2, 8.25, -4], [2, -0.75, 4]],
    #          [[2.5, -2, 1], [4, -1, 2], [1, 2, 4], [-1.5, 8, -1]],
    #          [[8, 2, -4], [2, 4, 0.], [4, 1, -2], [8, 4., 1]]],
    #         dtype=np.float64)
    #
    #     clf = MuComboClassifier()
    #     cost, label_score = clf._compute_cost(label_score, pred_classes, y, alphas,
    #                                           use_coop_coef=True)
    #
    #     np.testing.assert_almost_equal(label_score, expected_label_score,
    #                                    decimal)
    #
    #     label_score = np.array(
    #         [[[0, 0, np.log(4)], [np.log(8), 0, 0], [0, 0, 0], [0, 0, 0]],
    #          [[0, np.log(2), 0], [0, 0, 0], [0, 0, 0], [0, np.log(4), 0]],
    #          [[0, 0, 0], [np.log(8), 0, 0], [0, np.log(2), 0], [0, 0, 0]]],
    #         dtype=np.float64)
    #     alphas = np.array([np.log(2), np.log(4), np.log(8)])
    #     expected_label_score = np.array(
    #         [[[np.log(2), 0, np.log(4)],
    #           [np.log(8), 0, np.log(2)],
    #           [0, np.log(2), 0],
    #           [0, np.log(2), 0]],
    #          [[np.log(4), np.log(2), 0],
    #           [0, np.log(4), 0],
    #           [np.log(4), 0, 0],
    #           [np.log(4), np.log(4), 0]],
    #          [[0, 0, np.log(8)],
    #           [np.log(8), 0, np.log(8)],
    #           [np.log(8), np.log(2), 0],
    #           [0, np.log(8), 0]]],
    #         dtype=np.float64)
    #     expected_cost = np.array(
    #         [[[-2.5, 0.5, 2.], [4., 0.5, -4.5], [0.5, -1., 0.5], [1., 2., -3.]],
    #          [[-0.75, 0.5, 0.25], [1., 4., -5.], [4., -5., 1.], [4., 4., -8.]],
    #          [[-9., 1., 8.], [1., 0.125, -1.125], [4., -4.5, 0.5], [1., 8., -9.]]],
    #         dtype=np.float64)
    #
    #     clf = MuComboClassifier()
    #     cost, label_score = clf._compute_cost(label_score, pred_classes, y, alphas,
    #                                           use_coop_coef=False)
    #
    #     np.testing.assert_almost_equal(label_score, expected_label_score,
    #                                    decimal)
    #     np.testing.assert_almost_equal(cost, expected_cost, decimal)
    #
    #     label_score = np.array(
    #         [[[0, 0, np.log(4)], [np.log(8), 0, 0], [0, 0, 0], [0, 0, 0]],
    #          [[0, np.log(2), 0], [0, 0, 0], [0, 0, 0], [0, np.log(4), 0]],
    #          [[0, 0, 0], [np.log(8), 0, 0], [0, np.log(2), 0], [0, 0, 0]]],
    #         dtype=np.float64)
    #     alphas = np.array([np.log(2), np.log(4), np.log(8)])
    #     expected_label_score = np.array(
    #         [[[np.log(2), 0, np.log(4)],
    #           [np.log(8), 0, np.log(2)],
    #           [0, np.log(2), 0],
    #           [0, np.log(2), 0]],
    #          [[np.log(4), np.log(2), 0],
    #           [0, 0, 0],
    #           [0, 0, 0],
    #           [np.log(4), np.log(4), 0]],
    #          [[0, 0, 0],
    #           [np.log(8), 0, np.log(8)],
    #           [0, np.log(2), 0],
    #           [0, np.log(8), 0]]],
    #         dtype=np.float64)
    #     expected_cost = np.array(
    #         [[[-2.5, 0.5, 2.], [4., 0.5, -4.5], [0.5, -1., 0.5], [1., 2., -3.]],
    #          [[-0.75, 0.5, 0.25], [1., 1., -2.], [1., -2., 1.], [4., 4., -8.]],
    #          [[-2., 1., 1.], [1., 0.125, -1.125], [0.5, -1., 0.5], [1., 8., -9.]]],
    #         dtype=np.float64)
    #
    #     clf = MuComboClassifier()
    #     cost, label_score = clf._compute_cost(label_score, pred_classes, y, alphas,
    #                                           use_coop_coef=True)
    #
    #     np.testing.assert_almost_equal(label_score, expected_label_score,
    #                                    decimal)
    #     np.testing.assert_almost_equal(cost, expected_cost, decimal)
    #
    #
    # def test_algo_options():
    #     np.random.seed(seed)
    #
    #     n_estimators = 10
    #
    #     clf = MuComboClassifier(n_estimators=n_estimators, best_view_mode='edge')
    #     clf.fit(iris.data, iris.target, iris.views_ind)
    #     score = clf.score(iris.data, iris.target)
    #     assert_greater(score, 0.95, "Failed with score = {}".format(score))
    #
    #     clf = MuComboClassifier(n_estimators=n_estimators, best_view_mode='error')
    #     clf.fit(iris.data, iris.target, iris.views_ind)
    #     score = clf.score(iris.data, iris.target)
    #     assert_greater(score, 0.95, "Failed with score = {}".format(score))
    #
    #     assert_raises(ValueError, MuComboClassifier(), best_view_mode='test')
    #
    #     clf = MuComboClassifier()
    #     clf.best_view_mode = 'test'
    #     assert_raises(ValueError, clf.fit, iris.data, iris.target, iris.views_ind)
    #
    #

    def test_fit_views_ind(self):
       X = np.array([[1., 1., 1.], [-1., -1., -1.]])
       y = np.array([0, 1])
       expected_views_ind = np.array([0, 1, 3])
       clf = MuComboClassifier()
       clf.fit(X, y)
       # np.testing.assert_equal(clf.X_.views_ind, expected_views_ind)

    #     assert_array_equal(clf.views_ind_, expected_views_ind)
    # #
    def test_class_variation(self):
        # Check that classes labels can be integers or strings and can be stored
        # into any kind of sequence
        X = np.array([[1., 1., 1.], [-1., -1., -1.]])
        views_ind = np.array([0, 1, 3])
        y = np.array([3, 1])
        clf = MuComboClassifier()
        clf.fit(X, y, views_ind)
        np.testing.assert_almost_equal(clf.predict(X), y)

        y = np.array(["class_1", "class_2"])
        clf = MuComboClassifier()
        clf.fit(X, y)
        np.testing.assert_equal(clf.predict(X), y)
        # Check that misformed or inconsistent inputs raise expections
        X = np.zeros((5, 4, 2))
        y = np.array([0, 1])
        self.assertRaises(ValueError, clf.fit, X, y, views_ind)


    #     assert_raises(ValueError, clf.fit, X, y, views_ind)
    #
    #     X = ["str1", "str2"]
    #     assert_raises(ValueError, clf.fit, X, y, views_ind)
    #
    #     X = np.array([[1., 1., 1.], [-1., -1., -1.]])
    #     y = np.array([1])
    #     views_ind = np.array([0, 1, 3])
    #     clf = MuComboClassifier()
    #     assert_raises(ValueError, clf.fit, X, y, views_ind)
    #
    #     y = np.array([1, 0, 0, 1])
    #     views_ind = np.array([0, 1, 3])
    #     clf = MuComboClassifier()
    #     assert_raises(ValueError, clf.fit, X, y, views_ind)
    #
    #     y = np.array([3.2, 1.1])
    #     clf = MuComboClassifier()
    #     assert_raises(ValueError, clf.fit, X, y, views_ind)
    #
    #     y = np.array([0, 1])
    #     views_ind = np.array([0, 3, 1])
    #     clf = MuComboClassifier()
    #     assert_raises(ValueError, clf.fit, X, y, views_ind)
    #
    #     views_ind = np.array([-1, 1, 3])
    #     clf = MuComboClassifier()
    #     assert_raises(ValueError, clf.fit, X, y, views_ind)
    #
    #     views_ind = np.array([0, 1, 4])
    #     clf = MuComboClassifier()
    #     assert_raises(ValueError, clf.fit, X, y, views_ind)
    #
    #     views_ind = np.array([0.5, 1, 3])
    #     clf = MuComboClassifier()
    #     assert_raises(ValueError, clf.fit, X, y, views_ind)
    #
    #     views_ind = np.array("test")
    #     clf = MuComboClassifier()
    #     assert_raises(ValueError, clf.fit, X, y, views_ind)
    #
    #     views_ind = np.zeros((3, 2, 4))
    #     clf = MuComboClassifier()
    #     assert_raises(ValueError, clf.fit, X, y, views_ind)
    #
    #     views_ind = np.array([[-1], [1, 2]])
    #     clf = MuComboClassifier()
    #     assert_raises(ValueError, clf.fit, X, y, views_ind)
    #
    #     views_ind = np.array([[3], [1, 2]])
    #     clf = MuComboClassifier()
    #     assert_raises(ValueError, clf.fit, X, y, views_ind)
    #
    #     views_ind = np.array([[0.5], [1, 2]])
    #     clf = MuComboClassifier()
    #     assert_raises(ValueError, clf.fit, X, y, views_ind)
    #
    #     views_ind = np.array([[-1, 0], [1, 2]])
    #     clf = MuComboClassifier()
    #     assert_raises(ValueError, clf.fit, X, y, views_ind)
    #
    #     views_ind = np.array([[0, 3], [1, 2]])
    #     clf = MuComboClassifier()
    #     assert_raises(ValueError, clf.fit, X, y, views_ind)
    #
    #     views_ind = np.array([[0.5], [1], [2]])
    #     clf = MuComboClassifier()
    #     assert_raises(ValueError, clf.fit, X, y, views_ind)
    #
    #
    def test_decision_function(self):
        clf = MuComboClassifier()
        clf.fit(self.iris.data, self.iris.target, self.iris.views_ind)
        X = np.zeros((4, 3))
        with self.assertRaises(ValueError):
            clf.decision_function(X)
        X = self.iris.data[:, 0:15]
        dec = clf.decision_function(X)
        dec_expected = np.load(get_dataset_path("dec_iris.npy"))
        np.testing.assert_almost_equal(dec, dec_expected, 9)

    def test_predict(self):
        clf = MuComboClassifier()
        X = np.array([[0., 0.5, 0.7], [1., 1.5, 1.7], [2., 2.5, 2.7]])
        y = np.array([1, 1, 1])
        clf.fit(X, y)
        y_expected = clf.predict(X)
        np.testing.assert_almost_equal(y, y_expected, 9)


    # def test_limit_cases():
    #     np.random.seed(seed)
    #
    #     # Check that using empty data raises an exception
    #     X = np.array([[]])
    #     y = np.array([])
    #     clf = MuComboClassifier()
    #     assert_raises(ValueError, clf.fit, X, y)
    #
    #     # Check that fit() works for the smallest possible dataset
    #     X = np.array([[0.]])
    #     y = np.array([0])
    #     clf = MuComboClassifier()
    #     clf.fit(X, y)
    #     assert_array_equal(clf.predict(X), y)
    #     assert_array_equal(clf.predict(np.array([[1.]])), np.array([0]))
    #
    #     # Check that fit() works with samples from a single class
    #     X = np.array([[0., 0.5, 0.7], [1., 1.5, 1.7], [2., 2.5, 2.7]])
    #     y = np.array([1, 1, 1])
    #     views_ind = np.array([0, 1, 3])
    #     clf = MuComboClassifier()
    #     clf.fit(X, y, views_ind)
    #     assert_array_equal(clf.predict(X), y)
    #     assert_array_equal(clf.predict(np.array([[-1., 0., 1.]])), np.array([1]))
    #
    #     X = np.array([[0., 0.5, 0.7], [1., 1.5, 1.7], [2., 2.5, 2.7]])
    #     y = np.array([1, 1, 1])
    #     views_ind = np.array([[0, 2], [1]])
    #     clf = MuComboClassifier()
    #     clf.fit(X, y, views_ind)
    #     assert_array_equal(clf.predict(X), y)
    #     assert_array_equal(clf.predict(np.array([[-1., 0., 1.]])), np.array([1]))


    def test_simple_predict(self):
        #np.random.seed(seed)

        # Simple example with 2 classes and 1 view
        X = np.array(
            [[1.1, 2.1],
             [2.1, 0.2],
             [0.7, 1.2],
             [-0.9, -1.8],
             [-1.1, -2.2],
             [-0.3, -1.3]])
        y = np.array([0, 0, 0, 1, 1, 1])
        views_ind = np.array([0, 2])
        clf = MuComboClassifier()
        clf.fit(X, y, views_ind)
        #assert_array_equal(clf.predict(X), y)
        #assert_array_equal(clf.predict(np.array([[1., 1.], [-1., -1.]])),
        #                   np.array([0, 1]))
        #assert_equal(clf.decision_function(X).shape, y.shape)

        views_ind = np.array([[1, 0]])
        clf = MuComboClassifier()
        clf.fit(X, y, views_ind)
        np.testing.assert_almost_equal(clf.predict(X), y)



        #assert_array_equal(clf.predict(X), y)
        #assert_array_equal(clf.predict(np.array([[1., 1.], [-1., -1.]])),
        #                 np.array([0, 1]))
        #assert_equal(clf.decision_function(X).shape, y.shape)
    #
    #     # Simple example with 2 classes and 2 views
    #     X = np.array(
    #         [[1.1, 2.1, 0.5],
    #          [2.1, 0.2, 1.2],
    #          [0.7, 1.2, 2.1],
    #          [-0.9, -1.8, -0.3],
    #          [-1.1, -2.2, -0.9],
    #          [-0.3, -1.3, -1.4]])
    #     y = np.array([0, 0, 0, 1, 1, 1])
    #     views_ind = np.array([0, 2, 3])
    #     clf = MuComboClassifier()
    #     clf.fit(X, y, views_ind)
    #     assert_array_equal(clf.predict(X), y)
    #     assert_array_equal(clf.predict(np.array([[1., 1., 1.], [-1., -1., -1.]])),
    #                        np.array([0, 1]))
    #     assert_equal(clf.decision_function(X).shape, y.shape)
    #
    #     views_ind = np.array([[2, 0], [1]])
    #     clf = MuComboClassifier()
    #     clf.fit(X, y, views_ind)
    #     assert_array_equal(clf.predict(X), y)
    #     assert_array_equal(clf.predict(np.array([[1., 1., 1.], [-1., -1., -1.]])),
    #                        np.array([0, 1]))
    #     assert_equal(clf.decision_function(X).shape, y.shape)
    #
    #     # Simple example with 2 classes and 3 views
    #     X = np.array(
    #         [[1.1, 2.1, 0.5, 1.2, 1.7],
    #          [2.1, 0.2, 1.2, 0.6, 1.3],
    #          [0.7, 1.2, 2.1, 1.1, 0.9],
    #          [-0.9, -1.8, -0.3, -2.1, -1.1],
    #          [-1.1, -2.2, -0.9, -1.5, -1.2],
    #          [-0.3, -1.3, -1.4, -0.6, -0.7]])
    #     y = np.array([0, 0, 0, 1, 1, 1])
    #     views_ind = np.array([0, 2, 3, 5])
    #     clf = MuComboClassifier()
    #     clf.fit(X, y, views_ind)
    #     assert_array_equal(clf.predict(X), y)
    #     data = np.array([[1., 1., 1., 1., 1.], [-1., -1., -1., -1., -1.]])
    #     assert_array_equal(clf.predict(data), np.array([0, 1]))
    #     assert_equal(clf.decision_function(X).shape, y.shape)
    #
    #     views_ind = np.array([[2, 0], [1], [3, 4]])
    #     clf = MuComboClassifier()
    #     clf.fit(X, y, views_ind)
    #     assert_array_equal(clf.predict(X), y)
    #     data = np.array([[1., 1., 1., 1., 1.], [-1., -1., -1., -1., -1.]])
    #     assert_array_equal(clf.predict(data), np.array([0, 1]))
    #     assert_equal(clf.decision_function(X).shape, y.shape)
    #
    #     # Simple example with 3 classes and 3 views
    #     X = np.array(
    #         [[1.1, -1.2, 0.5, 1.2, -1.7],
    #          [2.1, -0.2, 0.9, 0.6, -1.3],
    #          [0.7, 1.2, 2.1, 1.1, 0.9],
    #          [0.9, 1.8, 2.2, 2.1, 1.1],
    #          [-1.1, -2.2, -0.9, -1.5, -1.2],
    #          [-0.3, -1.3, -1.4, -0.6, -0.7]])
    #     y = np.array([0, 0, 1, 1, 2, 2])
    #     views_ind = np.array([0, 2, 3, 5])
    #     clf = MuComboClassifier()
    #     clf.fit(X, y, views_ind)
    #     assert_array_equal(clf.predict(X), y)
    #     data = np.array(
    #         [[1., -1., 1., 1., -1.],
    #          [1., 1., 1., 1., 1.],
    #          [-1., -1., -1., -1., -1.]])
    #     assert_array_equal(clf.predict(data), np.array([0, 1, 2]))
    #     assert_equal(clf.decision_function(X).shape, (X.shape[0], 3))
    #
    #     views_ind = np.array([[1, 0], [2], [3, 4]])
    #     clf = MuComboClassifier()
    #     clf.fit(X, y, views_ind)
    #     assert_array_equal(clf.predict(X), y)
    #     data = np.array(
    #         [[1., -1., 1., 1., -1.],
    #          [1., 1., 1., 1., 1.],
    #          [-1., -1., -1., -1., -1.]])
    #     assert_array_equal(clf.predict(data), np.array([0, 1, 2]))
    #     assert_equal(clf.decision_function(X).shape, (X.shape[0], 3))
    #
    #
    # def test_generated_examples():
    #     def generate_data_in_orthotope(n_samples, limits):
    #         limits = np.array(limits)
    #         n_features = limits.shape[0]
    #         data = np.random.random((n_samples, n_features))
    #         data = (limits[:, 1]-limits[:, 0]) * data + limits[:, 0]
    #         return data
    #
    #     n_samples = 100
    #
    #     np.random.seed(seed)
    #     view_0 = np.concatenate(
    #         (generate_data_in_orthotope(n_samples, [[0., 1.], [0., 1.]]),
    #          generate_data_in_orthotope(n_samples, [[1., 2.], [0., 1.]])))
    #     view_1 = generate_data_in_orthotope(2*n_samples, [[0., 1.], [0., 1.]])
    #     X = np.concatenate((view_0, view_1), axis=1)
    #     y = np.zeros(2*n_samples, dtype=np.int64)
    #     y[n_samples:] = 1
    #     views_ind = np.array([0, 2, 4])
    #     clf = MuComboClassifier(n_estimators=1)
    #     clf.fit(X, y, views_ind)
    #     assert_equal(clf.score(X, y), 1.)
    #
    #     np.random.seed(seed)
    #     view_0 = np.concatenate(
    #         (generate_data_in_orthotope(n_samples, [[0., 1.], [0., 1.]]),
    #          generate_data_in_orthotope(n_samples, [[1., 2.], [0., 1.]]),
    #          generate_data_in_orthotope(n_samples, [[0., 1.], [0., 1.]]),
    #          generate_data_in_orthotope(n_samples, [[0., 1.], [1., 2.]])))
    #     view_1 = np.concatenate(
    #         (generate_data_in_orthotope(n_samples, [[1., 2.], [0., 1.]]),
    #          generate_data_in_orthotope(n_samples, [[0., 1.], [0., 1.]]),
    #          generate_data_in_orthotope(n_samples, [[0., 1.], [1., 2.]]),
    #          generate_data_in_orthotope(n_samples, [[0., 1.], [0., 1.]])))
    #     X = np.concatenate((view_0, view_1), axis=1)
    #     y = np.zeros(4*n_samples, dtype=np.int64)
    #     y[2*n_samples:] = 1
    #     views_ind = np.array([0, 2, 4])
    #     clf = MuComboClassifier(n_estimators=3)
    #     clf.fit(X, y, views_ind)
    #     assert_equal(clf.score(X, y), 1.)
    #
    #     np.random.seed(seed)
    #     view_0 = np.concatenate(
    #         (generate_data_in_orthotope(n_samples, [[0., 1.], [0., 1.]]),
    #          generate_data_in_orthotope(n_samples, [[1., 2.], [0., 1.]]),
    #          generate_data_in_orthotope(n_samples, [[0., 1.], [0., 1.]])))
    #     view_1 = np.concatenate(
    #         (generate_data_in_orthotope(n_samples, [[1., 2.], [0., 1.]]),
    #          generate_data_in_orthotope(n_samples, [[0., 1.], [0., 1.]]),
    #          generate_data_in_orthotope(n_samples, [[0., 1.], [0., 1.]])))
    #     view_2 = np.concatenate(
    #         (generate_data_in_orthotope(n_samples, [[0., 1.], [0., 1.]]),
    #          generate_data_in_orthotope(n_samples, [[0., 1.], [0., 1.]]),
    #          generate_data_in_orthotope(n_samples, [[1., 2.], [0., 1.]])))
    #     X = np.concatenate((view_0, view_1, view_2), axis=1)
    #     y = np.zeros(3*n_samples, dtype=np.int64)
    #     y[n_samples:2*n_samples] = 1
    #     y[2*n_samples:] = 2
    #     views_ind = np.array([0, 2, 4, 6])
    #     clf = MuComboClassifier(n_estimators=3)
    #     clf.fit(X, y, views_ind)
    #     assert_equal(clf.score(X, y), 1.)
    #
    #     np.random.seed(seed)
    #     view_0 = np.concatenate(
    #         (generate_data_in_orthotope(n_samples, [[0., 1.], [0., 1.]]),
    #          generate_data_in_orthotope(n_samples, [[1., 2.], [0., 1.]]),
    #          generate_data_in_orthotope(n_samples, [[0., 2.], [0., 1.]])))
    #     view_1 = np.concatenate(
    #         (generate_data_in_orthotope(n_samples, [[1., 2.], [0., 1.]]),
    #          generate_data_in_orthotope(n_samples, [[0., 2.], [0., 1.]]),
    #          generate_data_in_orthotope(n_samples, [[0., 1.], [0., 1.]])))
    #     view_2 = np.concatenate(
    #         (generate_data_in_orthotope(n_samples, [[0., 2.], [0., 1.]]),
    #          generate_data_in_orthotope(n_samples, [[0., 1.], [0., 1.]]),
    #          generate_data_in_orthotope(n_samples, [[1., 2.], [0., 1.]])))
    #     X = np.concatenate((view_0, view_1, view_2), axis=1)
    #     y = np.zeros(3*n_samples, dtype=np.int64)
    #     y[n_samples:2*n_samples] = 1
    #     y[2*n_samples:] = 2
    #     views_ind = np.array([0, 2, 4, 6])
    #     clf = MuComboClassifier(n_estimators=4)
    #     clf.fit(X, y, views_ind)
    #     assert_equal(clf.score(X, y), 1.)
    #

    def test_classifier(self):
        return check_estimator(MuComboClassifier())
    #
    #
    # def test_iris():
    #     # Check consistency on dataset iris.
    #
    #     np.random.seed(seed)
    #     n_estimators = 5
    #     classes = np.unique(iris.target)
    #
    #     for views_ind in [iris.views_ind, np.array([[0, 2], [1, 3]])]:
    #         clf = MuComboClassifier(n_estimators=n_estimators)
    #
    #         clf.fit(iris.data, iris.target, views_ind)
    #
    #         assert_true(np.all((0. <= clf.estimator_errors_)
    #                            & (clf.estimator_errors_ <= 1.)))
    #         assert_true(np.all(np.diff(clf.estimator_errors_) < 0.))
    #
    #         assert_array_equal(classes, clf.classes_)
    #         assert_equal(clf.decision_function(iris.data).shape[1], len(classes))
    #
    #         score = clf.score(iris.data, iris.target)
    #         assert_greater(score, 0.95, "Failed with score = {}".format(score))
    #
    #         assert_equal(len(clf.estimators_), n_estimators)
    #
    #         # Check for distinct random states
    #         assert_equal(len(set(est.random_state for est in clf.estimators_)),
    #                      len(clf.estimators_))


    def test_staged_methods(self):
        seed = 7
        n_estimators = 10
    #
        target_two_classes = np.zeros(self.iris.target.shape, dtype=np.int64)
        target_two_classes[target_two_classes.shape[0]//2:] = 1
    #
        data = (
               (self.iris.data, self.iris.target, self.iris.views_ind),
                (self.iris.data, self.iris.target, np.array([[0, 2], [1, 3]])),
                (self.iris.data, target_two_classes, self.iris.views_ind),
                (self.iris.data, target_two_classes, np.array([[0, 2], [1, 3]])),
               )

    #
        for X, y, views_ind in data:
            clf = MuComboClassifier(n_estimators=n_estimators, random_state=seed)
            clf.fit(X, y, views_ind)
            staged_dec_func = [dec_f for dec_f in clf.staged_decision_function(X)]
            staged_predict = [predict for predict in clf.staged_predict(X)]
            staged_score = [score for score in clf.staged_score(X, y)]
            self.assertEqual(len(staged_dec_func), n_estimators)
            self.assertEqual(len(staged_predict), n_estimators)
            self.assertEqual(len(staged_score), n_estimators)
            # assert_equal(len(staged_dec_func), n_estimators)
            # assert_equal(len(staged_predict), n_estimators)
            # assert_equal(len(staged_score), n_estimators)
    #
    #         for ind in range(n_estimators):
    #             clf = MuComboClassifier(n_estimators=ind+1, random_state=seed)
    #             clf.fit(X, y, views_ind)
    #             dec_func = clf.decision_function(X)
    #             predict = clf.predict(X)
    #             score = clf.score(X, y)
    #             assert_array_equal(dec_func, staged_dec_func[ind])
    #             assert_array_equal(predict, staged_predict[ind])
    #             assert_equal(score, staged_score[ind])
    #
    #
    def test_gridsearch(self):
    #     np.random.seed(seed)
    #
    #     # Check that base trees can be grid-searched.
        mumbo = MuComboClassifier(base_estimator=DecisionTreeClassifier())
        parameters = {'n_estimators': (1, 2),
                      'base_estimator__max_depth': (1, 2)}
        clf = GridSearchCV(mumbo, parameters)
        clf.fit(self.iris.data, self.iris.target, views_ind=self.iris.views_ind)
        self.assertEqual(clf.best_params_,{'base_estimator__max_depth': 2, 'n_estimators': 2})

        multimodal_data = MultiModalArray(self.iris.data, views_ind=self.iris.views_ind)
        clf = GridSearchCV(mumbo, parameters)
        clf.fit(multimodal_data, self.iris.target)

        self.assertEqual(clf.best_params_, {'base_estimator__max_depth': 2, 'n_estimators': 2})

    # def test_pick         le():
    #     np.random.seed(seed)
    #
    #     # Check pickability.
    #
    #     clf = MuComboClassifier()
    #     clf.fit(iris.data, iris.target, iris.views_ind)
    #     score = clf.score(iris.data, iris.target)
    #     dump = pickle.dumps(clf)
    #
    #     clf_loaded = pickle.loads(dump)
    #     assert_equal(type(clf_loaded), clf.__class__)
    #     score_loaded = clf_loaded.score(iris.data, iris.target)
    #     assert_equal(score, score_loaded)
    #
    #
    def test_base_estimator_score(self):
    #     np.random.seed(seed)
    #
        """ Test different base estimators."""
        n_estimators = 5
        clf = MuComboClassifier(RandomForestClassifier(), n_estimators=n_estimators)
        clf.fit(self.iris.data, self.iris.target, self.iris.views_ind)
        score = clf.score(self.iris.data, self.iris.target)
        self.assertGreater(score, 0.95, "Failed with score = {}".format(score))

        clf = MuComboClassifier(SVC(), n_estimators=n_estimators)
        clf.fit(self.iris.data, self.iris.target, self.iris.views_ind)
        score = clf.score(self.iris.data, self.iris.target)
        self.assertGreater(score, 0.95, "Failed with score = {}".format(score))

    #     # Check that using a base estimator that doesn't support sample_weight
    #     # raises an error.
        clf = MuComboClassifier(NoSampleWeightLasso())

        self.assertRaises(ValueError, clf.fit, self.iris.data, self.iris.target, self.iris.views_ind)
    #     assert_raises(ValueError, clf.fit, iris.data, iris.target, iris.views_ind)
    #
    #
    # def test_sparse_classification():
    #     # Check classification with sparse input.
    #
    #     np.random.seed(seed)
    #
    #     class CustomSVC(SVC):
    #         """SVC variant that records the nature of the training set."""
    #
    #         def fit(self, X, y, sample_weight=None):
    #             """Modification on fit caries data type for later verification."""
    #             super(CustomSVC, self).fit(X, y, sample_weight=sample_weight)
    #             self.data_type_ = type(X)
    #             return self
    #
    #     n_estimators = 5
    #     X_dense = iris.data
    #     y = iris.target
    #
    #     for sparse_format in [csc_matrix, csr_matrix, lil_matrix, coo_matrix,
    #                           dok_matrix]:
    #         for views_ind in (iris.views_ind, np.array([[0, 2], [1, 3]])):
    #             X_sparse = sparse_format(X_dense)
    #
    #             clf_sparse = MuComboClassifier(
    #                 base_estimator=CustomSVC(),
    #                 random_state=seed,
    #                 n_estimators=n_estimators)
    #             clf_sparse.fit(X_sparse, y, views_ind)
    #
    #             clf_dense = MuComboClassifier(
    #                 base_estimator=CustomSVC(),
    #                 random_state=seed,
    #                 n_estimators=n_estimators)
    #             clf_dense.fit(X_dense, y, views_ind)
    #
    #             assert_array_equal(clf_sparse.decision_function(X_sparse),
    #                                clf_dense.decision_function(X_dense))
    #
    #             assert_array_equal(clf_sparse.predict(X_sparse),
    #                                clf_dense.predict(X_dense))
    #
    #             assert_equal(clf_sparse.score(X_sparse, y),
    #                          clf_dense.score(X_dense, y))
    #
    #             for res_sparse, res_dense in \
    #                     zip(clf_sparse.staged_decision_function(X_sparse),
    #                         clf_dense.staged_decision_function(X_dense)):
    #                 assert_array_equal(res_sparse, res_dense)
    #
    #             for res_sparse, res_dense in \
    #                     zip(clf_sparse.staged_predict(X_sparse),
    #                         clf_dense.staged_predict(X_dense)):
    #                 assert_array_equal(res_sparse, res_dense)
    #
    #             for res_sparse, res_dense in \
    #                     zip(clf_sparse.staged_score(X_sparse, y),
    #                         clf_dense.staged_score(X_dense, y)):
    #                 assert_equal(res_sparse, res_dense)
    #
    #             # Check that sparsity of data is maintained during training
    #             types = [clf.data_type_ for clf in clf_sparse.estimators_]
    #             if sparse_format == csc_matrix:
    #                 assert_true(all([type_ == csc_matrix for type_ in types]))
    #             else:
    #                 assert_true(all([type_ == csr_matrix for type_ in types]))
    #

if __name__ == '__main__':
    unittest.main()
    # suite = unittest.TestLoader().loadTestsFromTestCase
    # (TestMuComboClassifier().test_class_variation())
    # unittest.TextTestRunner(verbosity=2).run(suite)