# -*- coding: utf-8 -*-
"""
==============
Use Case MumBo
==============
Use case for all classifier of multimodallearn MumBo

multi class digit from sklearn, multivue
 - vue 0 digit data (color of sklearn)
 - vue 1 gradiant of image in first direction
 - vue 2 gradiant of image in second direction

"""
from __future__ import absolute_import
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from multimodal.datasets.base import load_dict, save_dict
from multimodal.tests.data.get_dataset_path import get_dataset_path
from multimodal.datasets.data_sample import MultiModalArray

from multimodal.boosting.mumbo import MumboClassifier

from usecase_function import plot_subplot


if __name__ == '__main__':
    # file = get_dataset_path("digit_histogram.npy")
    file = get_dataset_path("digit_col_grad.npy")
    y = np.load(get_dataset_path("digit_y.npy"))
    base_estimator = DecisionTreeClassifier(max_depth=4)
    dic_digit = load_dict(file)
    XX =MultiModalArray(dic_digit)
    X_train, X_test, y_train, y_test = train_test_split(XX, y)

    est2 = MumboClassifier(base_estimator=base_estimator).fit(X_train, y_train)
    y_pred2 = est2.predict(X_test)
    y_pred22 = est2.predict(X_train)
    print("result of MumboClassifier on digit ")
    result2 = np.mean(y_pred2.ravel() == y_test.ravel()) * 100
    print(result2)

    fig = plt.figure(figsize=(12., 11.))
    fig.suptitle("Mumbo: result" + str(result2), fontsize=16)
    plot_subplot(X_train, y_train, y_pred22 , 0, (4, 1, 1), "train vue 0" )
    plot_subplot(X_test, y_test,y_pred2, 0, (4, 1, 2), "test vue 0" )
    plot_subplot(X_test, y_test, y_pred2, 1, (4, 1, 3), "test vue 1" )
    plot_subplot(X_test, y_test,y_pred2, 2, (4, 1, 4), "test vue 2" )
    # plt.legend()
    plt.show()
