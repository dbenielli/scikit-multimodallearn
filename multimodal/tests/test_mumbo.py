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
#"""Testing for the mumbo module."""#
# Author: Florent JAILLET - Laboratoire d'Informatique et Systèmes - UMR 7020

import pickle
import unittest
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, coo_matrix, dok_matrix
from scipy.sparse import lil_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.utils.estimator_checks import check_estimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from multimodal.boosting.mumbo import MumboClassifier

from multimodal.tests.test_combo import NoSampleWeightLasso

class TestMumboClassifier(unittest.TestCase):

    @classmethod
    def setUpClass(clf):
        # Load the iris dataset
        iris = datasets.load_iris()
        iris.views_ind = np.array([0, 2, 4])
        clf.iris = iris

    def test_sparse(self):
        rng = np.random.RandomState(0)
        X = rng.rand(40, 10)
        X[X < .8] = 0
        X_csr = csr_matrix(X)
        clf = MumboClassifier()
        y = (4 * rng.rand(40)).astype(int)
        clf.fit(X_csr, y)

    def test_init_var(self):
        n_classes = 3

        n_views = 3
        y = np.array([0, 2, 1, 2])
        expected_cost = np.array(
            [[[-2, 1, 1], [1, 1, -2], [1, -2, 1], [1, 1, -2]],
             [[-2, 1, 1], [1, 1, -2], [1, -2, 1], [1, 1, -2]],
             [[-2, 1, 1], [1, 1, -2], [1, -2, 1], [1, 1, -2]]],
            dtype=np.float64)
        expected_cost_glob = np.array(
            [[-2, 1, 1], [1, 1, -2], [1, -2, 1], [1, 1, -2]], dtype=np.float64)
        expected_label_score = np.zeros((n_views, y.shape[0], n_classes))
        expected_label_score_glob = np.zeros((y.shape[0], n_classes))
        expected_predicted_classes_shape = ((n_views, y.shape[0]))

        clf = MumboClassifier()
        clf.n_classes_ = n_classes
        (cost, cost_glob, label_score, label_score_glob,
         predicted_classes) = clf._init_var(n_views, y)
        np.testing.assert_equal(cost, expected_cost)

        np.testing.assert_equal(cost_glob, expected_cost_glob)
        np.testing.assert_equal(label_score, expected_label_score)
        np.testing.assert_equal(label_score_glob, expected_label_score_glob)
        self.assertEqual(predicted_classes.shape, expected_predicted_classes_shape)


    def test_compute_edge_global(self):
        cost_global = np.array([[-2, 1, 1], [1, 1, -2], [1, -2, 1], [1, 1, -2]],
                               dtype=np.float64)
        predicted_classes = np.array([[0, 0, 1, 1], [0, 1, 0, 2], [2, 2, 0, 0]])
        y = np.array([0, 2, 1, 2])
        expected_edge_global = np.array([0.25, 0.25, -0.125])

        clf = MumboClassifier()
        edge_global = clf._compute_edge_global(cost_global, predicted_classes, y)
        np.testing.assert_equal(edge_global, expected_edge_global)


    def test_compute_dist(self):
        cost = np.array(
            [[[-2, 1, 1], [-1, -1, -2], [1, -2, 1], [1, 1, -2]],
             [[-1, 2, 2], [2, 2, -1], [-2, 4, -2], [2, 2, -4]],
             [[1, 4, -4], [-1, 3, -1], [-2, 2, 4], [4, 4, -4]]],
            dtype=np.float64)
        y = np.array([0, 2, 1, 2])
        expected_dist = np.array(
            [[0.25, 0.25, 0.25, 0.25], [0.5, 0.5, -2., 2.], [-0.5, 0.5, -1., 2.]])

        clf = MumboClassifier()
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

    def test_compute_coop_coef(self):
        y = np.array([0, 1, 2, 0])
        predicted_classes = np.array([[0, 0, 1, 1], [0, 1, 0, 2], [2, 2, 0, 0]])
        expected_coop_coef = np.array([[1, 0, 1, 0], [1, 1, 1, 0], [0, 0, 1, 1]],
                                      dtype=np.float64)

        clf = MumboClassifier()
        coop_coef = clf._compute_coop_coef(predicted_classes, y)
        np.testing.assert_equal(coop_coef, expected_coop_coef)

    def test_compute_edges(self):
        cost = np.array(
            [[[-2, 1, 1], [-1, -1, -2], [1, -2, 1], [1, 1, -2]],
             [[-2, 2, 2], [2, 2, -4], [-2, -4, -2], [2, 2, -4]],
             [[1, 4, -4], [-1, 3, -1], [-2, 4, 4], [4, 4, -1]]],
            dtype=np.float64)
        predicted_classes = np.array([[0, 2, 1, 1], [0, 1, 0, 2], [2, 2, 0, 1]])
        y = np.array([0, 2, 1, 2])
        expected_edges = np.array([1.25, 0.75, 0.25])

        clf = MumboClassifier()
        edges = clf._compute_edges(cost, predicted_classes, y)
        np.testing.assert_equal(edges, expected_edges)

    def test_compute_alphas(self):
        decimal = 12
        expected_alpha = 0.5
        edge = (np.e-1.) / (np.e+1.)

        clf = MumboClassifier()
        alpha = clf._compute_alphas(edge)

        np.testing.assert_almost_equal(alpha, expected_alpha, decimal)

        expected_alphas = np.array([0.5, 1., 2.])
        tmp = np.array([np.e, np.e**2, np.e**4])
        edges = (tmp-1.) / (tmp+1.)

        alphas = clf._compute_alphas(edges)
        np.testing.assert_almost_equal(alphas, expected_alphas, decimal)

    def test_compute_cost_global(self):
        decimal = 12
        label_score_glob = np.array(
            [[-1, -2, 4], [-8, 1, 4], [2, 8, -4], [2, -1, 4]],
            dtype=np.float64)
        best_pred_classes = np.array([0, 1, 0, 2])
        y = np.array([0, 2, 1, 2])
        alpha = 0.5
        expected_label_score_glob = np.array(
            [[-0.5, -2, 4], [-8, 1.5, 4], [2.5, 8, -4], [2, -1, 4.5]],
            dtype=np.float64)

        clf = MumboClassifier()
        cost_glob, label_score_glob = clf._compute_cost_global(
            label_score_glob, best_pred_classes, y, alpha)
        np.testing.assert_almost_equal(label_score_glob, expected_label_score_glob,
                                       decimal)

        label_score_glob = np.zeros((4, 3), dtype=np.float64)
        alpha = 0.
        expected_label_score_glob = np.zeros((4, 3), dtype=np.float64)
        expected_cost_glob = np.array(
            [[-2, 1, 1], [1, 1, -2], [1, -2, 1], [1, 1, -2]],
            dtype=np.float64)

        cost_glob, label_score_glob = clf._compute_cost_global(
            label_score_glob, best_pred_classes, y, alpha)
        np.testing.assert_equal(label_score_glob, expected_label_score_glob)
        np.testing.assert_equal(cost_glob, expected_cost_glob, decimal)
        label_score_glob = np.array(
            [[0, 0, np.log(4)], [np.log(8), 0, 0], [0, 0, 0], [0, 0, 0]],
            dtype=np.float64)
        alpha = np.log(2)
        expected_label_score_glob = np.array(
            [[alpha, 0, np.log(4)],
             [np.log(8), alpha, 0],
             [alpha, 0, 0],
             [0, 0, alpha]],
            dtype=np.float64)
        expected_cost_glob = np.array(
            [[-2.5, 0.5, 2.], [8., 2., -10.], [2., -3., 1.], [0.5, 0.5, -1.]],
            dtype=np.float64)

        cost_glob, label_score_glob = clf._compute_cost_global(
            label_score_glob, best_pred_classes, y, alpha)

        np.testing.assert_almost_equal(label_score_glob, expected_label_score_glob,
                                       decimal)
        np.testing.assert_almost_equal(cost_glob, expected_cost_glob, decimal)

    def test_compute_cost(self):
        decimal = 12
        label_score = np.array(
            [[[-1, -2, 4], [-8, 1, 4], [2, 8, -4], [2, -1, 4]],
             [[2, -2, 1], [4, -1, 2], [1, 2, 4], [-2, 8, -1]],
             [[8, 2, -4], [2, 4, -2], [4, 1, -2], [8, 2, 1]]],
            dtype=np.float64)
        pred_classes = np.array([[0, 2, 1, 1], [0, 1, 0, 0], [2, 2, 0, 1]])
        y = np.array([0, 2, 1, 2])
        alphas = np.array([0.25, 0.5, 2.])
        expected_label_score = np.array(
            [[[-0.75, -2, 4], [-8, 1, 4.25], [2, 8.25, -4], [2, -0.75, 4]],
             [[2.5, -2, 1], [4, -0.5, 2], [1.5, 2, 4], [-1.5, 8, -1]],
             [[8, 2, -2.], [2, 4, 0.], [6., 1, -2], [8, 4., 1]]],
            dtype=np.float64)

        clf = MumboClassifier()
        cost, label_score = clf._compute_cost(label_score, pred_classes, y, alphas,
                                              use_coop_coef=False)

        np.testing.assert_almost_equal(label_score, expected_label_score,
                                       decimal)

        label_score = np.array(
            [[[-1, -2, 4], [-8, 1, 4], [2, 8, -4], [2, -1, 4]],
             [[2, -2, 1], [4, -1, 2], [1, 2, 4], [-2, 8, -1]],
             [[8, 2, -4], [2, 4, -2], [4, 1, -2], [8, 2, 1]]],
            dtype=np.float64)
        expected_label_score = np.array(
            [[[-0.75, -2, 4], [-8, 1, 4.25], [2, 8.25, -4], [2, -0.75, 4]],
             [[2.5, -2, 1], [4, -1, 2], [1, 2, 4], [-1.5, 8, -1]],
             [[8, 2, -4], [2, 4, 0.], [4, 1, -2], [8, 4., 1]]],
            dtype=np.float64)

        clf = MumboClassifier()
        cost, label_score = clf._compute_cost(label_score, pred_classes, y, alphas,
                                              use_coop_coef=True)

        np.testing.assert_almost_equal(label_score, expected_label_score,
                                       decimal)

        label_score = np.array(
            [[[0, 0, np.log(4)], [np.log(8), 0, 0], [0, 0, 0], [0, 0, 0]],
             [[0, np.log(2), 0], [0, 0, 0], [0, 0, 0], [0, np.log(4), 0]],
             [[0, 0, 0], [np.log(8), 0, 0], [0, np.log(2), 0], [0, 0, 0]]],
            dtype=np.float64)
        alphas = np.array([np.log(2), np.log(4), np.log(8)])
        expected_label_score = np.array(
            [[[np.log(2), 0, np.log(4)],
              [np.log(8), 0, np.log(2)],
              [0, np.log(2), 0],
              [0, np.log(2), 0]],
             [[np.log(4), np.log(2), 0],
              [0, np.log(4), 0],
              [np.log(4), 0, 0],
              [np.log(4), np.log(4), 0]],
             [[0, 0, np.log(8)],
              [np.log(8), 0, np.log(8)],
              [np.log(8), np.log(2), 0],
              [0, np.log(8), 0]]],
            dtype=np.float64)
        expected_cost = np.array(
            [[[-2.5, 0.5, 2.], [4., 0.5, -4.5], [0.5, -1., 0.5], [1., 2., -3.]],
             [[-0.75, 0.5, 0.25], [1., 4., -5.], [4., -5., 1.], [4., 4., -8.]],
             [[-9., 1., 8.], [1., 0.125, -1.125], [4., -4.5, 0.5], [1., 8., -9.]]],
            dtype=np.float64)

        clf = MumboClassifier()
        cost, label_score = clf._compute_cost(label_score, pred_classes, y, alphas,
                                              use_coop_coef=False)

        np.testing.assert_almost_equal(label_score, expected_label_score,
                                       decimal)
        np.testing.assert_almost_equal(cost, expected_cost, decimal)

        label_score = np.array(
            [[[0, 0, np.log(4)], [np.log(8), 0, 0], [0, 0, 0], [0, 0, 0]],
             [[0, np.log(2), 0], [0, 0, 0], [0, 0, 0], [0, np.log(4), 0]],
             [[0, 0, 0], [np.log(8), 0, 0], [0, np.log(2), 0], [0, 0, 0]]],
            dtype=np.float64)
        alphas = np.array([np.log(2), np.log(4), np.log(8)])
        expected_label_score = np.array(
            [[[np.log(2), 0, np.log(4)],
              [np.log(8), 0, np.log(2)],
              [0, np.log(2), 0],
              [0, np.log(2), 0]],
             [[np.log(4), np.log(2), 0],
              [0, 0, 0],
              [0, 0, 0],
              [np.log(4), np.log(4), 0]],
             [[0, 0, 0],
              [np.log(8), 0, np.log(8)],
              [0, np.log(2), 0],
              [0, np.log(8), 0]]],
            dtype=np.float64)
        expected_cost = np.array(
            [[[-2.5, 0.5, 2.], [4., 0.5, -4.5], [0.5, -1., 0.5], [1., 2., -3.]],
             [[-0.75, 0.5, 0.25], [1., 1., -2.], [1., -2., 1.], [4., 4., -8.]],
             [[-2., 1., 1.], [1., 0.125, -1.125], [0.5, -1., 0.5], [1., 8., -9.]]],
            dtype=np.float64)

        clf = MumboClassifier()
        cost, label_score = clf._compute_cost(label_score, pred_classes, y, alphas,
                                              use_coop_coef=True)

        np.testing.assert_almost_equal(label_score, expected_label_score,
                                       decimal)
        np.testing.assert_almost_equal(cost, expected_cost, decimal)

    def test_algo_options(self):
        seed = 7
        np.random.seed(seed)

        n_estimators = 10
        #print("iris views ind", self.iris.views_ind)
        clf = MumboClassifier(n_estimators=n_estimators, best_view_mode='edge')
        clf.fit(self.iris.data, self.iris.target, self.iris.views_ind)
        score = clf.score(self.iris.data, self.iris.target)
        self.assertGreater(score, 0.95, "Failed with score = {}".format(score))

        clf = MumboClassifier(n_estimators=n_estimators, best_view_mode='error')
        clf.fit(self.iris.data, self.iris.target, self.iris.views_ind)
        score = clf.score(self.iris.data, self.iris.target)
        self.assertGreater(score, 0.95, "Failed with score = {}".format(score))

        self.assertRaises(ValueError, MumboClassifier, best_view_mode='test')

        clf = MumboClassifier()
        clf.best_view_mode = 'test'
        self.assertRaises(ValueError, clf.fit, self.iris.data,
                          self.iris.target, self.iris.views_ind)

    def test_fit_arg(self):
        seed = 7
        np.random.seed(seed)

        # Check that using the default value for views_ind corresponds to using 2
        # views
        X = np.array([[1., 1., 1.], [-1., -1., -1.]])
        y = np.array([0, 1])
        expected_views_ind = np.array([0, 1, 3])
        clf = MumboClassifier()
        clf.fit(X, y)
        np.testing.assert_equal(clf.X_.views_ind, expected_views_ind)

        # Check that classes labels can be integers or strings and can be stored
        # into any kind of sequence
        views_ind = np.array([0, 1, 3])
        y = np.array([3, 1])
        clf = MumboClassifier()
        clf.fit(X, y, views_ind)
        np.testing.assert_equal(clf.predict(X), y)

        y = np.array(["class_1", "class_2"])
        clf = MumboClassifier()
        clf.fit(X, y)
        np.testing.assert_equal(clf.predict(X), y)

        y = [1, 0]
        clf = MumboClassifier()
        clf.fit(X, y, views_ind)
        np.testing.assert_equal(clf.predict(X), y)

        y = (2, 1)
        clf = MumboClassifier()
        clf.fit(X, y, views_ind)
        np.testing.assert_equal(clf.predict(X), y)

        # Check that misformed or inconsistent inputs raise expections
        X = np.zeros((5, 4, 2))
        y = np.array([0, 1])
        self.assertRaises(ValueError, clf.fit, X, y, views_ind)

        X = ["str1", "str2"]
        self.assertRaises(ValueError, clf.fit, X, y, views_ind)

        X = np.array([[1., 1., 1.], [-1., -1., -1.]])
        y = np.array([1])
        views_ind = np.array([0, 1, 3])
        clf = MumboClassifier()
        self.assertRaises(ValueError, clf.fit, X, y, views_ind)

        y = np.array([1, 0, 0, 1])
        views_ind = np.array([0, 1, 3])
        clf = MumboClassifier()
        self.assertRaises(ValueError, clf.fit, X, y, views_ind)

        y = np.array([3.2, 1.1])
        clf = MumboClassifier()
        self.assertRaises(ValueError, clf.fit, X, y, views_ind)

        y = np.array([0, 1])
        views_ind = np.array([0, 3, 1])
        clf = MumboClassifier()
        self.assertRaises(ValueError, clf.fit, X, y, views_ind)

        views_ind = np.array([-1, 1, 3])
        clf = MumboClassifier()
        self.assertRaises(ValueError, clf.fit, X, y, views_ind)

        views_ind = np.array([0, 1, 4])
        clf = MumboClassifier()
        self.assertRaises(ValueError, clf.fit, X, y, views_ind)

        views_ind = np.array([0.5, 1, 3])
        clf = MumboClassifier()
        self.assertRaises(ValueError, clf.fit, X, y, views_ind)

        views_ind = np.array("test")
        clf = MumboClassifier()
        self.assertRaises(ValueError, clf.fit, X, y, views_ind)

        views_ind = np.zeros((3, 2, 4))
        clf = MumboClassifier()
        self.assertRaises(ValueError, clf.fit, X, y, views_ind)

        views_ind = np.array([np.array([-1]), np.array([1, 2])], dtype=object)
        clf = MumboClassifier()
        self.assertRaises(ValueError, clf.fit, X, y, views_ind)

        views_ind = np.array([np.array([3]), np.array([1, 2])], dtype=object)
        clf = MumboClassifier()
        self.assertRaises(ValueError, clf.fit, X, y, views_ind)

        views_ind = np.array([np.array([0.5]), np.array([1, 2])], dtype=object)
        clf = MumboClassifier()
        self.assertRaises(ValueError, clf.fit, X, y, views_ind)

        views_ind = np.array([[-1, 0], [1, 2]])
        clf = MumboClassifier()
        self.assertRaises(ValueError, clf.fit, X, y, views_ind)

        views_ind = np.array([[0, 3], [1, 2]])
        clf = MumboClassifier()
        self.assertRaises(ValueError, clf.fit, X, y, views_ind)

        views_ind = np.array([[0.5], [1], [2]])
        clf = MumboClassifier()
        self.assertRaises(ValueError, clf.fit, X, y, views_ind)

    def test_decision_function_arg(self):
        # Test that decision_function() gives proper exception on deficient input.
        seed = 7
        np.random.seed(seed)

        clf = MumboClassifier()
        clf.fit(self.iris.data, self.iris.target, self.iris.views_ind)

        X = np.zeros((4, 3))
        self.assertRaises(ValueError, clf.decision_function, X)
        X = np.zeros((4, 5))
        self.assertRaises(ValueError, clf.decision_function, X)
        X = np.zeros((5, 4, 2))
        self.assertRaises(ValueError, clf.decision_function, X)
        X = ["str1", "str2"]
        self.assertRaises(ValueError, clf.decision_function, X)

    def test_limit_cases(self):
        seed = 7
        np.random.seed(seed)

        # Check that using empty data raises an exception
        X = np.array([[]])
        y = np.array([])
        clf = MumboClassifier()
        self.assertRaises(ValueError, clf.fit, X, y)

        # Check that fit() works for the smallest possible dataset
        X = np.array([[0.]])
        y = np.array([0])
        clf = MumboClassifier()
        clf.fit(X, y)
        np.testing.assert_equal(clf.predict(X), y)
        np.testing.assert_equal(clf.predict(np.array([[1.]])), np.array([0]))

        # Check that fit() works with samples from a single class
        X = np.array([[0., 0.5, 0.7], [1., 1.5, 1.7], [2., 2.5, 2.7]])
        y = np.array([1, 1, 1])
        views_ind = np.array([0, 1, 3])
        clf = MumboClassifier()
        clf.fit(X, y, views_ind)
        np.testing.assert_equal(clf.predict(X), y)
        np.testing.assert_equal(clf.predict(np.array([[-1., 0., 1.]])), np.array([1]))

        X = np.array([[0., 0.5, 0.7], [1., 1.5, 1.7], [2., 2.5, 2.7]])
        y = np.array([1, 1, 1])
        views_ind = np.array([np.array([0, 2]), np.array([1])], dtype=object)
        clf = MumboClassifier()
        clf.fit(X, y, views_ind)
        np.testing.assert_equal(clf.predict(X), y)
        np.testing.assert_equal(clf.predict(np.array([[-1., 0., 1.]])), np.array([1]))

    def test_simple_examples(self):
        seed =7
        np.random.seed(seed)

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
        clf = MumboClassifier()
        clf.fit(X, y, views_ind)
        np.testing.assert_equal(clf.predict(X), y)
        np.testing.assert_equal(clf.predict(np.array([[1., 1.], [-1., -1.]])),
                           np.array([0, 1]))
        X = clf._global_X_transform(X, clf.X_.views_ind)
        self.assertEqual(clf.decision_function(X).shape, y.shape)

        views_ind = np.array([[1, 0]])
        clf = MumboClassifier()
        clf.fit(X, y, views_ind)
        np.testing.assert_equal(clf.predict(X), y)
        np.testing.assert_equal(clf.predict(np.array([[1., 1.], [-1., -1.]])),
                           np.array([0, 1]))
        self.assertEqual(clf.decision_function(X).shape, y.shape)

        # Simple example with 2 classes and 2 views
        X = np.array(
            [[1.1, 2.1, 0.5],
             [2.1, 0.2, 1.2],
             [0.7, 1.2, 2.1],
             [-0.9, -1.8, -0.3],
             [-1.1, -2.2, -0.9],
             [-0.3, -1.3, -1.4]])
        y = np.array([0, 0, 0, 1, 1, 1])
        views_ind = np.array([0, 2, 3])
        clf = MumboClassifier()
        clf.fit(X, y, views_ind)
        np.testing.assert_equal(clf.predict(X), y)
        np.testing.assert_equal(clf.predict(np.array([[1., 1., 1.], [-1., -1., -1.]])),
                           np.array([0, 1]))
        self.assertEqual(clf.decision_function(X).shape, y.shape)

        views_ind = np.array([np.array([2, 0]), np.array([1])], dtype=object)
        clf = MumboClassifier()
        clf.fit(X, y, views_ind)
        np.testing.assert_equal(clf.predict(X), y)
        np.testing.assert_equal(clf.predict(np.array([[1., 1., 1.], [-1., -1., -1.]])),
                           np.array([0, 1]))
        self.assertEqual(clf.decision_function(X).shape, y.shape)

        # Simple example with 2 classes and 3 views
        X = np.array(
            [[1.1, 2.1, 0.5, 1.2, 1.7],
             [2.1, 0.2, 1.2, 0.6, 1.3],
             [0.7, 1.2, 2.1, 1.1, 0.9],
             [-0.9, -1.8, -0.3, -2.1, -1.1],
             [-1.1, -2.2, -0.9, -1.5, -1.2],
             [-0.3, -1.3, -1.4, -0.6, -0.7]])
        y = np.array([0, 0, 0, 1, 1, 1])
        views_ind = np.array([0, 2, 3, 5])
        clf = MumboClassifier()
        clf.fit(X, y, views_ind)
        np.testing.assert_equal(clf.predict(X), y)
        data = np.array([[1., 1., 1., 1., 1.], [-1., -1., -1., -1., -1.]])
        np.testing.assert_equal(clf.predict(data), np.array([0, 1]))
        self.assertEqual(clf.decision_function(X).shape, y.shape)

        views_ind = np.array([np.array([2, 0]), np.array([1]), np.array([3, 4])], dtype=object)
        clf = MumboClassifier()
        clf.fit(X, y, views_ind)
        np.testing.assert_equal(clf.predict(X), y)
        data = np.array([[1., 1., 1., 1., 1.], [-1., -1., -1., -1., -1.]])
        np.testing.assert_equal(clf.predict(data), np.array([0, 1]))
        self.assertEqual(clf.decision_function(X).shape, y.shape)

        # Simple example with 3 classes and 3 views
        X = np.array(
            [[1.1, -1.2, 0.5, 1.2, -1.7],
             [2.1, -0.2, 0.9, 0.6, -1.3],
             [0.7, 1.2, 2.1, 1.1, 0.9],
             [0.9, 1.8, 2.2, 2.1, 1.1],
             [-1.1, -2.2, -0.9, -1.5, -1.2],
             [-0.3, -1.3, -1.4, -0.6, -0.7]])
        y = np.array([0, 0, 1, 1, 2, 2])
        views_ind = np.array([0, 2, 3, 5])
        clf = MumboClassifier()
        clf.fit(X, y, views_ind)
        np.testing.assert_equal(clf.predict(X), y)
        data = np.array(
            [[1., -1., 1., 1., -1.],
             [1., 1., 1., 1., 1.],
             [-1., -1., -1., -1., -1.]])
        np.testing.assert_equal(clf.predict(data), np.array([0, 1, 2]))
        self.assertEqual(clf.decision_function(X).shape, (X.shape[0], 3))

        views_ind = np.array([np.array([1, 0]), np.array([2]), np.array([3, 4])], dtype=object)
        clf = MumboClassifier()
        clf.fit(X, y, views_ind)
        np.testing.assert_equal(clf.predict(X), y)
        data = np.array(
            [[1., -1., 1., 1., -1.],
             [1., 1., 1., 1., 1.],
             [-1., -1., -1., -1., -1.]])
        np.testing.assert_equal(clf.predict(data), np.array([0, 1, 2]))
        self.assertEqual(clf.decision_function(X).shape, (X.shape[0], 3))

    def test_generated_examples(self):
        seed = 7
        def generate_data_in_orthotope(n_samples, limits):
            limits = np.array(limits)
            n_features = limits.shape[0]
            data = np.random.random((n_samples, n_features))
            data = (limits[:, 1]-limits[:, 0]) * data + limits[:, 0]
            return data

        n_samples = 100

        np.random.seed(seed)
        view_0 = np.concatenate(
            (generate_data_in_orthotope(n_samples, [[0., 1.], [0., 1.]]),
             generate_data_in_orthotope(n_samples, [[1., 2.], [0., 1.]])))
        view_1 = generate_data_in_orthotope(2*n_samples, [[0., 1.], [0., 1.]])
        X = np.concatenate((view_0, view_1), axis=1)
        y = np.zeros(2*n_samples, dtype=np.int64)
        y[n_samples:] = 1
        views_ind = np.array([0, 2, 4])
        clf = MumboClassifier(n_estimators=1)
        clf.fit(X, y, views_ind)
        self.assertEqual(clf.score(X, y), 1.)

        np.random.seed(seed)
        view_0 = np.concatenate(
            (generate_data_in_orthotope(n_samples, [[0., 1.], [0., 1.]]),
             generate_data_in_orthotope(n_samples, [[1., 2.], [0., 1.]]),
             generate_data_in_orthotope(n_samples, [[0., 1.], [0., 1.]]),
             generate_data_in_orthotope(n_samples, [[0., 1.], [1., 2.]])))
        view_1 = np.concatenate(
            (generate_data_in_orthotope(n_samples, [[1., 2.], [0., 1.]]),
             generate_data_in_orthotope(n_samples, [[0., 1.], [0., 1.]]),
             generate_data_in_orthotope(n_samples, [[0., 1.], [1., 2.]]),
             generate_data_in_orthotope(n_samples, [[0., 1.], [0., 1.]])))
        X = np.concatenate((view_0, view_1), axis=1)
        y = np.zeros(4*n_samples, dtype=np.int64)
        y[2*n_samples:] = 1
        views_ind = np.array([0, 2, 4])
        clf = MumboClassifier(n_estimators=3)
        clf.fit(X, y, views_ind)
        self.assertEqual(clf.score(X, y), 1.)

        np.random.seed(seed)
        view_0 = np.concatenate(
            (generate_data_in_orthotope(n_samples, [[0., 1.], [0., 1.]]),
             generate_data_in_orthotope(n_samples, [[1., 2.], [0., 1.]]),
             generate_data_in_orthotope(n_samples, [[0., 1.], [0., 1.]])))
        view_1 = np.concatenate(
            (generate_data_in_orthotope(n_samples, [[1., 2.], [0., 1.]]),
             generate_data_in_orthotope(n_samples, [[0., 1.], [0., 1.]]),
             generate_data_in_orthotope(n_samples, [[0., 1.], [0., 1.]])))
        view_2 = np.concatenate(
            (generate_data_in_orthotope(n_samples, [[0., 1.], [0., 1.]]),
             generate_data_in_orthotope(n_samples, [[0., 1.], [0., 1.]]),
             generate_data_in_orthotope(n_samples, [[1., 2.], [0., 1.]])))
        X = np.concatenate((view_0, view_1, view_2), axis=1)
        y = np.zeros(3*n_samples, dtype=np.int64)
        y[n_samples:2*n_samples] = 1
        y[2*n_samples:] = 2
        views_ind = np.array([0, 2, 4, 6])
        clf = MumboClassifier(n_estimators=3)
        clf.fit(X, y, views_ind)
        self.assertEqual(clf.score(X, y), 1.)

        np.random.seed(seed)
        view_0 = np.concatenate(
            (generate_data_in_orthotope(n_samples, [[0., 1.], [0., 1.]]),
             generate_data_in_orthotope(n_samples, [[1., 2.], [0., 1.]]),
             generate_data_in_orthotope(n_samples, [[0., 2.], [0., 1.]])))
        view_1 = np.concatenate(
            (generate_data_in_orthotope(n_samples, [[1., 2.], [0., 1.]]),
             generate_data_in_orthotope(n_samples, [[0., 2.], [0., 1.]]),
             generate_data_in_orthotope(n_samples, [[0., 1.], [0., 1.]])))
        view_2 = np.concatenate(
            (generate_data_in_orthotope(n_samples, [[0., 2.], [0., 1.]]),
             generate_data_in_orthotope(n_samples, [[0., 1.], [0., 1.]]),
             generate_data_in_orthotope(n_samples, [[1., 2.], [0., 1.]])))
        X = np.concatenate((view_0, view_1, view_2), axis=1)
        y = np.zeros(3*n_samples, dtype=np.int64)
        y[n_samples:2*n_samples] = 1
        y[2*n_samples:] = 2
        views_ind = np.array([0, 2, 4, 6])
        clf = MumboClassifier(n_estimators=4)
        clf.fit(X, y, views_ind)
        self.assertEqual(clf.score(X, y), 1.)


    def test_classifier(self):
        # X_zero_features = np.empty(0).reshape(3, 0)
        # y = np.array([1, 0, 1])
        # e = MumboClassifier()
        # e.fit(X_zero_features, y)
        # print(e.predict(X_zero_features))
        return check_estimator(MumboClassifier())

    def test_iris(self):
        # Check consistency on dataset iris.
        seed = 7
        np.random.seed(seed)
        n_estimators = 5
        classes = np.unique(self.iris.target)

        for views_ind in [self.iris.views_ind, np.array([[0, 2], [1, 3]])]:
            clf = MumboClassifier(n_estimators=n_estimators)

            clf.fit(self.iris.data, self.iris.target, views_ind)

            self.assertTrue(np.all((0. <= clf.estimator_errors_)
                               & (clf.estimator_errors_ <= 1.)))
            self.assertTrue(np.all(np.diff(clf.estimator_errors_) < 0.))

            np.testing.assert_equal(classes, clf.classes_)
            self.assertEqual(clf.decision_function(self.iris.data).shape[1], len(classes))

            score = clf.score(self.iris.data, self.iris.target)
            self.assertGreater(score, 0.95, "Failed with score = {}".format(score))

            self.assertEqual(len(clf.estimators_), n_estimators)

            # Check for distinct random states
            self.assertEqual(len(set(est.random_state for est in clf.estimators_)),
                         len(clf.estimators_))

    def test_staged_methods(self):
        n_estimators = 10
        seed = 7

        target_two_classes = np.zeros(self.iris.target.shape, dtype=np.int64)
        target_two_classes[target_two_classes.shape[0]//2:] = 1

        data = (
                (self.iris.data, self.iris.target, self.iris.views_ind),
                (self.iris.data, self.iris.target, np.array([[0, 2], [1, 3]])),
                (self.iris.data, target_two_classes, self.iris.views_ind),
                (self.iris.data, target_two_classes, np.array([[0, 2], [1, 3]])),
               )

        for X, y, views_ind in data:
            clf = MumboClassifier(n_estimators=n_estimators, random_state=seed)
            clf.fit(X, y, views_ind)
            staged_dec_func = [dec_f for dec_f in clf.staged_decision_function(X)]
            staged_predict = [predict for predict in clf.staged_predict(X)]
            staged_score = [score for score in clf.staged_score(X, y)]

            self.assertEqual(len(staged_dec_func), n_estimators)
            self.assertEqual(len(staged_predict), n_estimators)
            self.assertEqual(len(staged_score), n_estimators)

            for ind in range(n_estimators):
                clf = MumboClassifier(n_estimators=ind+1, random_state=seed)
                clf.fit(X, y, views_ind)
                dec_func = clf.decision_function(X)
                predict = clf.predict(X)
                score = clf.score(X, y)
                np.testing.assert_equal(dec_func, staged_dec_func[ind])
                np.testing.assert_equal(predict, staged_predict[ind])
                self.assertEqual(score, staged_score[ind])

    def test_gridsearch(self):
        seed = 7
        np.random.seed(seed)

        # Check that base trees can be grid-searched.
        mumbo = MumboClassifier(base_estimator=DecisionTreeClassifier())
        parameters = {'n_estimators': (1, 2),
                      'base_estimator__max_depth': (1, 2)}
        clf = GridSearchCV(mumbo, parameters)
        clf.fit(self.iris.data, self.iris.target, views_ind=self.iris.views_ind)


    def test_pickle(self):
        seed = 7
        np.random.seed(seed)

        # Check pickability.

        clf = MumboClassifier()
        clf.fit(self.iris.data, self.iris.target, self.iris.views_ind)
        score = clf.score(self.iris.data, self.iris.target)
        dump = pickle.dumps(clf)
        clf_loaded = pickle.loads(dump)
        self.assertEqual(type(clf_loaded), clf.__class__)
        score_loaded = clf_loaded.score(self.iris.data, self.iris.target)
        self.assertEqual(score, score_loaded)

    def test_base_estimator(self):
        seed = 7
        np.random.seed(seed)

        # Test different base estimators.
        n_estimators = 5
        clf = MumboClassifier(RandomForestClassifier(), n_estimators=n_estimators)
        clf.fit(self.iris.data, self.iris.target, self.iris.views_ind)
        score = clf.score(self.iris.data, self.iris.target)
        self.assertGreater(score, 0.95, "Failed with score = {}".format(score))

        clf = MumboClassifier(SVC(), n_estimators=n_estimators)
        clf.fit(self.iris.data, self.iris.target, self.iris.views_ind)
        score = clf.score(self.iris.data, self.iris.target)
        self.assertGreater(score, 0.95, "Failed with score = {}".format(score))

        # Check that using a base estimator that doesn't support sample_weight
        # raises an error.
        clf = MumboClassifier(NoSampleWeightLasso())
        self.assertRaises(ValueError, clf.fit, self.iris.data, self.iris.target, self.iris.views_ind)


    def test_sparse_classification(self):
        # Check classification with sparse input.
        seed = 7
        np.random.seed(seed)

        class CustomSVC(SVC):
            """SVC variant that records the nature of the training set."""

            def fit(self, X, y, sample_weight=None):
                """Modification on fit caries data type for later verification."""
                super(CustomSVC, self).fit(X, y, sample_weight=sample_weight)
                self.data_type_ = type(X)
                return self

        n_estimators = 5
        X_dense = self.iris.data
        y = self.iris.target

        for sparse_format in [csc_matrix, csr_matrix]: #, lil_matrix, coo_matrix,dok_matrix]:
            for views_ind in (self.iris.views_ind, np.array([[0, 2], [1, 3]])):
                X_sparse = sparse_format(X_dense)
                clf_sparse = MumboClassifier(
                    base_estimator=CustomSVC(),
                    random_state=seed,
                    n_estimators=n_estimators)
                clf_sparse.fit(X_sparse, y, views_ind)

                clf_dense = MumboClassifier(
                    base_estimator=CustomSVC(),
                    random_state=seed,
                    n_estimators=n_estimators)
                clf_dense.fit(X_dense, y, views_ind)

                np.testing.assert_equal(clf_sparse.decision_function(X_sparse),
                                   clf_dense.decision_function(X_dense))

                np.testing.assert_equal(clf_sparse.predict(X_sparse),
                                   clf_dense.predict(X_dense))

                self.assertEqual(clf_sparse.score(X_sparse, y),
                             clf_dense.score(X_dense, y))

                for res_sparse, res_dense in \
                        zip(clf_sparse.staged_decision_function(X_sparse),
                            clf_dense.staged_decision_function(X_dense)):
                    np.testing.assert_equal(res_sparse, res_dense)

                for res_sparse, res_dense in \
                        zip(clf_sparse.staged_predict(X_sparse),
                            clf_dense.staged_predict(X_dense)):
                    np.testing.assert_equal(res_sparse, res_dense)

                for res_sparse, res_dense in \
                        zip(clf_sparse.staged_score(X_sparse, y),
                            clf_dense.staged_score(X_dense, y)):
                    np.testing.assert_equal(res_sparse, res_dense)

                # Check that sparsity of data is maintained during training
                types = [clf.data_type_ for clf in clf_sparse.estimators_]
                if sparse_format == csc_matrix:
                    self.assertTrue(all([issubclass(type_, csc_matrix)  for type_ in types]))
                else:
                    self.assertTrue(all([issubclass(type_, csr_matrix) for type_ in types]))

    def test_validate_X_predict(self):
        clf = MumboClassifier()
        X = np.random.randint(1, 10, (2, 10))
        y = [1, 0]
        clf.fit(X, y)
        X_pred = np.random.randint(1, 10, 10)
        self.assertRaises(ValueError, clf._validate_X_predict, X_pred)
        X_pred = np.random.randint(1,10,9)
        self.assertRaises(ValueError, clf._validate_X_predict, X_pred)
        X_pred = np.random.randint(1, 10, (2, 9))
        self.assertRaises(ValueError, clf._validate_X_predict, X_pred)


if __name__ == '__main__':
    unittest.main()
