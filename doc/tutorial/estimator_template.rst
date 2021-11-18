
.. _estim-template:

Estimator template
==================

To add a multimodal estimator based on the groundwork of scikit-multimodallearn,
please feel free to use the following template, while complying with the
`Developer's Guide <http://scikit-learn.org/stable/developers>`_ of the
scikit-learn project to ensure full compatibility.



.. code-block:: default

    import numpy as np
    from sklearn.base import ClassifierMixin, BaseEstimator
    from sklearn.utils import check_X_y
    from sklearn.utils.multiclass import check_classification_targets
    from sklearn.utils.validation import check_is_fitted
    from multimodal.boosting.boost import UBoosting


    class NewMultiModalEstimator(BaseEstimator, ClassifierMixin, UBoosting):
        r""""
        Your documentation
        """

        def __init__(self, your_attributes=None, ):
            self.your_attributes = your_attributes

        def fit(self, X, y, views_ind=None):
            """Build a multimodal classifier from the training set (X, y).

            Parameters
            ----------
            X : dict dictionary with all views
                or
                `MultiModalData` ,  `MultiModalArray`, `MultiModalSparseArray`
                or
                {array-like, sparse matrix}, shape = (n_samples, n_features)
                Training multi-view input samples.
                Sparse matrix can be CSC, CSR, COO, DOK, or LIL.
                COO, DOK and LIL are converted to CSR.

            y : array-like, shape = (n_samples,)
                Target values (class labels).

            views_ind : array-like (default=[0, n_features//2, n_features])
                Paramater specifying how to extract the data views from X:

                - If views_ind is a 1-D array of sorted integers, the entries
                  indicate the limits of the slices used to extract the views,
                  where view ``n`` is given by
                  ``X[:, views_ind[n]:views_ind[n+1]]``.

                  With this convention each view is therefore a view (in the NumPy
                  sense) of X and no copy of the data is done.

                - If views_ind is an array of arrays of integers, then each array
                  of integers ``views_ind[n]`` specifies the indices of the view
                  ``n``, which is then given by ``X[:, views_ind[n]]``.

                  With this convention each view creates therefore a partial copy
                  of the data in X. This convention is thus more flexible but less
                  efficient than the previous one.

            Returns
            -------
            self : object
                Returns self.

            Raises
            ------
            ValueError  estimator must support sample_weight

            ValueError where `X` and `view_ind` are not compatibles
            """

            # _global_X_transform processes the multimodal dataset to transform the
            # in the MultiModalArray format.
            self.X_ = self._global_X_transform(X, views_ind=views_ind)

            # Ensure proper format for views_ind and return number of views.
            views_ind_, n_views = self.X_._validate_views_ind(self.X_.views_ind,
                                                              self.X_.shape[1])

            # According to scikit learn guidelines.
            check_X_y(self.X_, y)
            if not isinstance(y, np.ndarray):
                y = np.asarray(y)
            check_classification_targets(y)
            self._validate_estimator()

            return self


        def predict(self, X):
            """Predict classes for X.

            Parameters
            ----------
            X : {array-like, sparse matrix}, shape = (n_samples, n_features)
                Multi-view input samples.
                Sparse matrix can be CSC, CSR, COO, DOK, or LIL.
                COO, DOK and LIL are converted to CSR.

            Returns
            -------
            y : numpy.ndarray, shape = (n_samples,)
                Predicted classes.

            Raises
            ------
            ValueError   'X' input matrix must be have the same total number of features
                         of 'X' fit data
            """
            # According to scikit learn guidelines
            check_is_fitted(self, ("your_attributes"))

            # _global_X_transform processes the multimodal dataset to transform the
            # in the MultiModalArray format.
            X = self._global_X_transform(X, views_ind=self.X_.views_ind)

            # Ensure that X is in the proper format.
            X = self._validate_X_predict(X)

            # Returning fake multi-class labels
            return np.random.randint(0, 5, size=X.shape[0])