Installation and development
============================

Dependencies
------------

**scikit-multimodallearn** works with **Python 3.5 or later**.

**scikit-multimodallearn** depends on **scikit-learn** (version >= 0.19) and **cvxopt**.

Optionally, **matplotlib** is required when running the examples.

Installation
------------

**scikit-multimodallearn* is
`available on PyPI <https://pypi.org/project/scikit-multimodallearn/>`_
and can be installed using **pip**::

  pip install multimodalboost

If you prefer to install directly from the **source code**, clone the **Git**
repository of the project and run the **setup.py** file with the following
commands::

  git clone git@gitlab.lis-lab.fr:dev/scikit-multimodallearn.git
  cd multimodalboost
  python setup.py install

or alternatively use **pip**::

  pip install git+https://gitlab.lis-lab.fr/dev/scikit-multimodallearn.git

Development
-----------

The development of scikit-multimodallearn follows the guidelines provided by the
scikit-learn community.

Refer to the `Developer's Guide <http://scikit-learn.org/stable/developers>`_
of the scikit-learn project for general details. Expanding the library can be
done by following the template provided in :ref:`estim-template` .

Source code
-----------

You can get the **source code** from the **Git** repository of the project::

  git clone git@gitlab.lis-lab.fr:dev/scikit-multimodallearn.git


Testing
-------

**pytest** and **pytest-cov** are required to run the **test suite** with::

  pytest

A code coverage report is displayed in the terminal when running the tests.
An HTML version of the report is also stored in the directory **htmlcov**.

Generating the documentation
----------------------------

The generation of the documentation requires **sphinx**, **sphinx-gallery**,
**numpydoc** and **matplotlib** and can be run with::

  python setup.py build_sphinx

The resulting files are stored in the directory **build/sphinx/html**.
