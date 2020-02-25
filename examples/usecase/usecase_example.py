# -*- coding: utf-8 -*-
"""
========
Use Case
========
Use case for all classifier of multimodallearn (in file mvml.py) is intended to be used with very simple simulated dataset

multi class digit from sklearn, multivue
 - vue 0 digit data (color of sklearn)
 - vue 1 gradiant of image in first direction
 - vue 2 gradiant of image in second direction

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib._color_data as mcd
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from multimodal.datasets.base import load_dict, save_dict
from multimodal.tests.data.get_dataset_path import get_dataset_path
from multimodal.datasets.data_sample import MultiModalArray
from multimodal.kernels.mvml import MVML
from multimodal.kernels.lpMKL import MKL
from multimodal.boosting.mumbo import MumboClassifier
from multimodal.boosting.cumbo import MuCumboClassifier

def plot_subplot(X, Y, Y_pred, vue, subplot, title):
    cn = mcd.CSS4_COLORS
    classes = np.unique(Y)
    n_classes = len(np.unique(Y))
    axs = plt.subplot(subplot[0],subplot[1],subplot[2])
    axs.set_title(title)
    #plt.scatter(X._extract_view(vue), X._extract_view(vue), s=40, c='gray',
    #            edgecolors=(0, 0, 0))
    for index, k in zip(range(n_classes), cn.keys()):
         Y_class, = np.where(Y==classes[index])
         Y_class_pred = np.intersect1d(np.where(Y_pred==classes[index])[0], np.where(Y_pred==Y)[0])
         plt.scatter(X._extract_view(vue)[Y_class],
                     X._extract_view(vue)[Y_class],
                     s=40, c=cn[k], edgecolors='blue', linewidths=2, label="class real class: "+str(index)) #
         plt.scatter(X._extract_view(vue)[Y_class_pred],
                     X._extract_view(vue)[Y_class_pred],
                     s=160, edgecolors='orange', linewidths=2, label="class prediction: "+str(index))


if __name__ == '__main__':
    # file = get_dataset_path("digit_histogram.npy")
    file = get_dataset_path("digit_col_grad.npy")
    y = np.load(get_dataset_path("digit_y.npy"))
    base_estimator = DecisionTreeClassifier(max_depth=4)
    dic_digit = load_dict(file)
    XX =MultiModalArray(dic_digit)
    X_train, X_test, y_train, y_test = train_test_split(XX, y)
    est1 = OneVsOneClassifier(MVML(lmbda=0.1, eta=1, nystrom_param=0.2)).fit(X_train, y_train)
    y_pred1 = est1.predict(X_test)
    y_pred11 = est1.predict(X_train)
    print("result of MVML on digit with oneversone")
    result1 = np.mean(y_pred1.ravel() == y_test.ravel()) * 100
    print(result1)
    est2 = MumboClassifier(base_estimator=base_estimator).fit(X_train, y_train)
    y_pred2 = est2.predict(X_test)
    y_pred22 = est2.predict(X_train)
    print("result of MumboClassifier on digit ")
    result2 = np.mean(y_pred2.ravel() == y_test.ravel()) * 100
    print(result2)

    est3 = MuCumboClassifier(base_estimator=base_estimator).fit(X_train, y_train)
    y_pred3 = est3.predict(X_test)
    y_pred33 = est3.predict(X_train)
    print("result of MuCumboClassifier on digit ")
    result3 = np.mean(y_pred3.ravel() == y_test.ravel()) * 100
    print(result3)

    est4 = OneVsOneClassifier(MKL(lmbda=0.1, nystrom_param=0.2)).fit(X_train, y_train)
    y_pred4 = est4.predict(X_test)
    y_pred44 = est4.predict(X_train)
    print("result of MKL on digit with oneversone")
    result4 = np.mean(y_pred4.ravel() == y_test.ravel()) * 100
    print(result4)

    fig = plt.figure(figsize=(12., 11.))
    fig.suptitle("MKL : result" + str(result4), fontsize=16)
    plot_subplot(X_train, y_train, y_pred44  ,0, (4, 1, 1), "train vue 0" )
    plot_subplot(X_test, y_test,y_pred4 , 0, (4, 1, 2), "test vue 0" )
    plot_subplot(X_test, y_test, y_pred4,1, (4, 1, 3), "test vue 1" )
    plot_subplot(X_test, y_test,y_pred4, 2, (4, 1, 4), "test vue 2" )
    # plt.legend()
    plt.show()
    fig = plt.figure(figsize=(12., 11.))
    fig.suptitle("MuCumbo: result" + str(result3), fontsize=16)
    plot_subplot(X_train, y_train, y_pred33  ,0, (4, 1, 1), "train vue 0" )
    plot_subplot(X_test, y_test,y_pred3 , 0, (4, 1, 2), "test vue 0" )
    plot_subplot(X_test, y_test, y_pred3,1, (4, 1, 3), "test vue 1" )
    plot_subplot(X_test, y_test,y_pred3, 2, (4, 1, 4), "test vue 2" )
    # plt.legend()
    plt.show()
    fig = plt.figure(figsize=(12., 11.))
    fig.suptitle("Mumbo: result" + str(result2), fontsize=16)
    plot_subplot(X_train, y_train, y_pred22 , 0, (4, 1, 1), "train vue 0" )
    plot_subplot(X_test, y_test,y_pred2, 0, (4, 1, 2), "test vue 0" )
    plot_subplot(X_test, y_test, y_pred2, 1, (4, 1, 3), "test vue 1" )
    plot_subplot(X_test, y_test,y_pred2, 2, (4, 1, 4), "test vue 2" )
    # plt.legend()
    plt.show()
    fig = plt.figure(figsize=(12., 11.))
    fig.suptitle("MVML: result" + str(result1), fontsize=16)
    plot_subplot(X_train, y_train, y_pred11
                 , 0, (4, 1, 1), "train vue 0" )
    plot_subplot(X_test, y_test,y_pred1, 0, (4, 1, 2), "test vue 0" )
    plot_subplot(X_test, y_test, y_pred1, 1, (4, 1, 3), "test vue 1" )
    plot_subplot(X_test, y_test,y_pred1, 2, (4, 1, 4), "test vue 2" )
    #plt.legend()
    plt.show()
    #mvml = MVML(lmbda=0.1, eta=1, nystrom_param=0.2)
    #mvml.fit(dic_digit_histo, y)
