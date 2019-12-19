Installation and development
============================

Dependencies
------------

**multimodalboost** works with **Python 3.5 or later**.

**multimodalboost** depends on **scikit-learn** (version >= 0.19).

Optionally, **matplotlib** is required when running the examples.

Installation
------------

**multimodalboost** is
`available on PyPI <https://pypi.org/project/multimodalboost/>`_
and can be installed using **pip**::

  pip install multimodalboost

If you prefer to install directly from the **source code**, clone the **Git**
repository of the project and run the **setup.py** file with the following
commands::

  git clone git@gitlab.lis-lab.fr:dev/multimodalboost.git
  cd multimodalboost
  python setup.py install

or alternatively use **pip**::

  pip install git+https://gitlab.lis-lab.fr/dev/multimodalboost.git

Development
-----------

The development of multimodalboost follows the guidelines provided by the
scikit-learn community.

Refer to the `Developer's Guide <http://scikit-learn.org/stable/developers>`_
of the scikit-learn project for more details.

Source code
-----------

You can get the **source code** from the **Git** repository of the project::

  git clone git@gitlab.lis-lab.fr:dev/multimodalboost.git


Testing
-------

**pytest** and **pytest-cov** are required to run the **test suite** with::

  cd multimodalboost
  pytest

A code coverage report is displayed in the terminal when running the tests.
An HTML version of the report is also stored in the directory **htmlcov**.

Generating the documentation
----------------------------

The generation of the documentation requires **sphinx**, **sphinx-gallery**,
**numpydoc** and **matplotlib** and can be run with::

  python setup.py build_sphinx

The resulting files are stored in the directory **build/sphinx/html**.
