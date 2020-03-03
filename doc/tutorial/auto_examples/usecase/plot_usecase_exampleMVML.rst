.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_tutorial_auto_examples_usecase_plot_usecase_exampleMVML.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_tutorial_auto_examples_usecase_plot_usecase_exampleMVML.py:


=========================
Use Case of MVML on digit
========================
Use case for all classifier of multimodallearn MVML

multi class digit from sklearn, multivue
 - vue 0 digit data (color of sklearn)
 - vue 1 gradiant of image in first direction
 - vue 2 gradiant of image in second direction




.. image:: /tutorial/auto_examples/usecase/images/sphx_glr_plot_usecase_exampleMVML_001.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    WARNING:root:warning appears during fit process{'precond_A': 4, 'precond_A_1': 5}
    WARNING:root:warning appears during fit process{'precond_A': 4, 'precond_A_1': 2}
    WARNING:root:warning appears during fit process{'precond_A': 4, 'precond_A_1': 5}
    WARNING:root:warning appears during fit process{'precond_A': 4, 'precond_A_1': 4}
    WARNING:root:warning appears during fit process{'precond_A': 4, 'precond_A_1': 6}
    WARNING:root:warning appears during fit process{'precond_A': 4, 'precond_A_1': 5}
    WARNING:root:warning appears during fit process{'precond_A': 5, 'precond_A_1': 5}
    WARNING:root:warning appears during fit process{'precond_A': 4, 'precond_A_1': 6}
    WARNING:root:warning appears during fit process{'precond_A': 4, 'precond_A_1': 5}
    WARNING:root:warning appears during fit process{'precond_A': 3, 'precond_A_1': 4}
    WARNING:root:warning appears during fit process{'precond_A': 4, 'precond_A_1': 4}
    WARNING:root:warning appears during fit process{'precond_A': 3, 'precond_A_1': 5}
    WARNING:root:warning appears during fit process{'precond_A': 4, 'precond_A_1': 4}
    WARNING:root:warning appears during fit process{'precond_A': 4, 'precond_A_1': 6}
    WARNING:root:warning appears during fit process{'precond_A': 4, 'precond_A_1': 6}
    WARNING:root:warning appears during fit process{'precond_A': 4, 'precond_A_1': 5}
    WARNING:root:warning appears during fit process{'precond_A': 4, 'precond_A_1': 6}
    WARNING:root:warning appears during fit process{'precond_A': 4, 'precond_A_1': 4}
    WARNING:root:warning appears during fit process{'precond_A': 4, 'precond_A_1': 1}
    WARNING:root:warning appears during fit process{'precond_A': 4, 'precond_A_1': 2}
    WARNING:root:warning appears during fit process{'precond_A': 4, 'precond_A_1': 2}
    WARNING:root:warning appears during fit process{'precond_A': 4, 'precond_A_1': 1}
    WARNING:root:warning appears during fit process{'precond_A': 4, 'precond_A_1': 2}
    WARNING:root:warning appears during fit process{'precond_A': 4, 'precond_A_1': 5}
    WARNING:root:warning appears during fit process{'precond_A': 3, 'precond_A_1': 5}
    WARNING:root:warning appears during fit process{'precond_A': 3, 'precond_A_1': 4}
    WARNING:root:warning appears during fit process{'precond_A': 4, 'precond_A_1': 5}
    WARNING:root:warning appears during fit process{'precond_A': 3, 'precond_A_1': 4}
    WARNING:root:warning appears during fit process{'precond_A': 4, 'precond_A_1': 6}
    WARNING:root:warning appears during fit process{'precond_A': 4, 'precond_A_1': 5}
    WARNING:root:warning appears during fit process{'precond_A': 4, 'precond_A_1': 5}
    WARNING:root:warning appears during fit process{'precond_A': 3, 'precond_A_1': 4}
    WARNING:root:warning appears during fit process{'precond_A': 4, 'precond_A_1': 5}
    WARNING:root:warning appears during fit process{'precond_A': 3, 'precond_A_1': 2}
    WARNING:root:warning appears during fit process{'precond_A': 5, 'precond_A_1': 5}
    WARNING:root:warning appears during fit process{'precond_A': 4, 'precond_A_1': 5}
    WARNING:root:warning appears during fit process{'precond_A': 5, 'precond_A_1': 5}
    WARNING:root:warning appears during fit process{'precond_A': 4, 'precond_A_1': 4}
    WARNING:root:warning appears during fit process{'precond_A': 4, 'precond_A_1': 6}
    WARNING:root:warning appears during fit process{'precond_A': 4, 'precond_A_1': 4}
    WARNING:root:warning appears during fit process{'precond_A': 4, 'precond_A_1': 4}
    WARNING:root:warning appears during fit process{'precond_A': 4, 'precond_A_1': 6}
    WARNING:root:warning appears during fit process{'precond_A': 4, 'precond_A_1': 1}
    WARNING:root:warning appears during fit process{'precond_A': 4, 'precond_A_1': 5}
    WARNING:root:warning appears during fit process{'precond_A': 4, 'precond_A_1': 5}
    result of MVML on digit with oneversone
    96.88888888888889
    /home/dominique/projets/ANR-Lives/scikit-multimodallearn/examples/usecase/plot_usecase_exampleMVML.py:70: UserWarning: Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.
      plt.show()






|


.. code-block:: default


    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.multiclass import OneVsOneClassifier
    from sklearn.model_selection import train_test_split
    from multimodal.datasets.base import load_dict, save_dict
    from multimodal.tests.data.get_dataset_path import get_dataset_path
    from multimodal.datasets.data_sample import MultiModalArray
    from multimodal.kernels.mvml import MVML
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



.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 1 minutes  14.485 seconds)


.. _sphx_glr_download_tutorial_auto_examples_usecase_plot_usecase_exampleMVML.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: plot_usecase_exampleMVML.py <plot_usecase_exampleMVML.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: plot_usecase_exampleMVML.ipynb <plot_usecase_exampleMVML.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
