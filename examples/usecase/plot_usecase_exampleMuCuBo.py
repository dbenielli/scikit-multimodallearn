# -*- coding: utf-8 -*-
"""
=========================
Use Case MuCumBo on digit
=========================
Use case for all classifier of multimodallearn  MuCumBo

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

from multimodal.boosting.cumbo import MuCumboClassifier
from usecase_function import plot_subplot


if __name__ == '__main__':
    # file = get_dataset_path("digit_histogram.npy")
    file = get_dataset_path("digit_col_grad.npy")
    y = np.load(get_dataset_path("digit_y.npy"))
    base_estimator = DecisionTreeClassifier(max_depth=4)
    dic_digit = load_dict(file)
    XX =MultiModalArray(dic_digit)
    X_train, X_test, y_train, y_test = train_test_split(XX, y)
    est3 = MuCumboClassifier(base_estimator=base_estimator).fit(X_train, y_train)
    y_pred3 = est3.predict(X_test)
    y_pred33 = est3.predict(X_train)
    print("result of MuCumboClassifier on digit ")
    result3 = np.mean(y_pred3.ravel() == y_test.ravel()) * 100
    print(result3)

    fig = plt.figure(figsize=(12., 11.))
    fig.suptitle("MuCumbo: result" + str(result3), fontsize=16)
    plot_subplot(X_train, y_train, y_pred33  ,0, (4, 1, 1), "train vue 0 color" )
    plot_subplot(X_test, y_test,y_pred3 , 0, (4, 1, 2), "test vue 0 color" )
    plot_subplot(X_test, y_test, y_pred3,1, (4, 1, 3), "test vue 1 gradiant 0" )
    plot_subplot(X_test, y_test,y_pred3, 2, (4, 1, 4), "test vue 2 gradiant 1" )
    # plt.legend()
    plt.show()