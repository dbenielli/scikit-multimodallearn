.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_tutorial_auto_examples_usecase_plot_usecase_exampleMuCuBo.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_tutorial_auto_examples_usecase_plot_usecase_exampleMuCuBo.py:


=========================
Use Case MuCumBo on digit
=========================
Use case for all classifier of multimodallearn  MuCumBo

multi class digit from sklearn, multivue
 - vue 0 digit data (color of sklearn)
 - vue 1 gradiant of image in first direction
 - vue 2 gradiant of image in second direction




.. image:: /tutorial/auto_examples/usecase/images/sphx_glr_plot_usecase_exampleMuCuBo_001.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    result of MuCumboClassifier on digit 
    87.77777777777777
    /home/dominique/projets/ANR-Lives/scikit-multimodallearn/examples/usecase/plot_usecase_exampleMuCuBo.py:71: UserWarning: Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.
      plt.show()






|


.. code-block:: default


    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from multimodal.datasets.base import load_dict, save_dict
    from multimodal.tests.data.get_dataset_path import get_dataset_path
    from multimodal.datasets.data_sample import MultiModalArray

    from multimodal.boosting.cumbo import MuCumboClassifier
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

.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  14.171 seconds)


.. _sphx_glr_download_tutorial_auto_examples_usecase_plot_usecase_exampleMuCuBo.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: plot_usecase_exampleMuCuBo.py <plot_usecase_exampleMuCuBo.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: plot_usecase_exampleMuCuBo.ipynb <plot_usecase_exampleMuCuBo.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
