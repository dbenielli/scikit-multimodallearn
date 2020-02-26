# -*- coding: utf-8 -*-
"""
========================
Use Case Function module
========================

Function plot_subplot

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib._color_data as mcd


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
    pass