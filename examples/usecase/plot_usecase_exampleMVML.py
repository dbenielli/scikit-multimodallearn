# -*- coding: utf-8 -*-
"""
=========================
Use Case of MVML on digit
========================
Use case for all classifier of multimodallearn MVML

multi class digit from sklearn, multivue
 - vue 0 digit data (color of sklearn)
 - vue 1 gradiant of image in first direction
 - vue 2 gradiant of image in second direction

"""
from __future__ import absolute_import
import numpy as np
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split
from multimodal.datasets.base import load_dict, save_dict
from multimodal.tests.data.get_dataset_path import get_dataset_path
from multimodal.datasets.data_sample import MultiModalArray
from multimodal.kernels.mvml import MVML
from examples.usecase.usecase_function import plot_subplot


if __name__ == '__main__':
    # file = get_dataset_path("digit_histogram.npy")
    file = get_dataset_path("digit_col_grad.npy")
    y = np.load(get_dataset_path("digit_y.npy"))
    dic_digit = load_dict(file)
    XX =MultiModalArray(dic_digit)
    X_train, X_test, y_train, y_test = train_test_split(XX, y)
    est1 = OneVsOneClassifier(MVML(lmbda=0.1, eta=1, nystrom_param=0.2)).fit(X_train, y_train)
    y_pred1 = est1.predict(X_test)
    y_pred11 = est1.predict(X_train)
    print("result of MVML on digit with oneversone")
    result1 = np.mean(y_pred1.ravel() == y_test.ravel()) * 100
    print(result1)

    fig = plt.figure(figsize=(12., 11.))
    fig.suptitle("MVML: result" + str(result1), fontsize=16)
    plot_subplot(X_train, y_train, y_pred11
                 , 0, (4, 1, 1), "train vue 0 color" )
    plot_subplot(X_test, y_test,y_pred1, 0, (4, 1, 2), "test vue 0 color" )
    plot_subplot(X_test, y_test, y_pred1, 1, (4, 1, 3), "test vue 1 gradiant 0" )
    plot_subplot(X_test, y_test,y_pred1, 2, (4, 1, 4), "test vue 2 gradiant 1" )
    #plt.legend()
    plt.show()

