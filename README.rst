.. image:: https://gitlab.lis-lab.fr/dev/scikit-multimodallearn/badges/master/pipeline.svg
    :target: https://gitlab.lis-lab.fr/dev/scikit-multimodallearn/badges/master
    :alt: pipeline status
    
.. image:: https://gitlab.lis-lab.fr/dev/scikit-multimodallearn/badges/master/coverage.svg
    :target: https://gitlab.lis-lab.fr/dev/scikit-multimodallearn/badges/master
    :alt: coverage report


scikit-multimodallearn
======================

**scikit-multimodallearn** is a Python package implementing algorithms multimodal data.

It is compatible with `scikit-learn <http://scikit-learn.org/>`_, a popular
package for machine learning in Python.


Documentation
-------------

The **documentation** including installation instructions, API documentation
and examples is
`available online <http://dev.pages.lis-lab.fr/scikit-multimodallearn>`_.


Installation
------------

Dependencies
~~~~~~~~~~~~

**scikit-multimodallearn** works with **Python 3.5 or later**.

**scikit-multimodallearn** depends on **scikit-learn** (version 1.2.1).

Optionally, **matplotlib** is required to run the examples.

Installation using pip
~~~~~~~~~~~~~~~~~~~~~~

**scikit-multimodallearn** is
`available on PyPI <https://pypi.org/project/scikit-multimodallearn/>`_
and can be installed using **pip**::

  pip install scikit-multimodallearn


Development
-----------

The development of this package follows the guidelines provided by the
scikit-learn community.

Refer to the `Developer's Guide <http://scikit-learn.org/stable/developers>`_
of the scikit-learn project for more details.

Source code
~~~~~~~~~~~

You can get the **source code** from the **Git** repository of the project::

  git clone git@gitlab.lis-lab.fr:dev/scikit-multimodallearn.git

Testing
~~~~~~~

**pytest** and **pytest-cov** are required to run the **test suite** with::

  cd multimodal
  pytest

A code coverage report is displayed in the terminal when running the tests.
An HTML version of the report is also stored in the directory **htmlcov**.


Generating the documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The generation of the documentation requires **sphinx**, **sphinx-gallery**,
**numpydoc** and **matplotlib** and can be run with::

  python setup.py build_sphinx

The resulting files are stored in the directory **build/sphinx/html**.


Credits
-------

**scikit-multimodallearn** is developped by the
`development team <https://developpement.lis-lab.fr/>`_ of the
`LIS <http://www.lis-lab.fr/>`_.

If you use **scikit-multimodallearn** in a scientific publication, please cite the
following paper::

 @InProceedings{Koco:2011:BAMCC,
  author={Ko\c{c}o, Sokol and Capponi, C{\'e}cile},
  editor={Gunopulos, Dimitrios and Hofmann, Thomas and Malerba, Donato
          and Vazirgiannis, Michalis},
  title={A Boosting Approach to Multiview Classification with Cooperation},
  booktitle={Proceedings of the 2011 European Conference on Machine Learning
             and Knowledge Discovery in Databases - Volume Part II},
  year={2011},
  location={Athens, Greece},
  publisher={Springer-Verlag},
  address={Berlin, Heidelberg},
  pages={209--228},
  numpages = {20},
  isbn={978-3-642-23783-6}
  url={https://link.springer.com/chapter/10.1007/978-3-642-23783-6_14},
  keywords={boosting, classification, multiview learning,
            supervised learning},
 }

 @InProceedings{Huu:2019:BAMCC,
  author={Huusari, Riika, Kadri Hachem and Capponi, C{\'e}cile},
  editor={},
  title={Multi-view Metric Learning in Vector-valued Kernel Spaces},
  booktitle={arXiv:1803.07821v1},
  year={2018},
  location={Athens, Greece},
  publisher={},
  address={},
  pages={209--228},
  numpages = {12}
  isbn={978-3-642-23783-6}
  url={https://link.springer.com/chapter/10.1007/978-3-642-23783-6_14},
  keywords={boosting, classification, multiview learning,
            merric learning, vector-valued, kernel spaces},
 }

References
~~~~~~~~~~
* Sokol Koço, Cécile Capponi,
  `"Learning from Imbalanced Datasets with cross-view cooperation"`
  Linking and mining heterogeneous an multi-view data, Unsupervised and
  semi-supervised learning Series Editor M. Emre Celeri, pp 161-182, Springer

* Sokol Koço, Cécile Capponi,
  `"A boosting approach to multiview classification with cooperation"
  <https://link.springer.com/chapter/10.1007/978-3-642-23783-6_14>`_,
  Proceedings of the 2011 European Conference on Machine Learning (ECML),
  Athens, Greece, pp.209-228, 2011, Springer-Verlag.

* Sokol Koço,
  `"Tackling the uneven views problem with cooperation based ensemble
  learning methods" <http://www.theses.fr/en/2013AIXM4101>`_,
  PhD Thesis, Aix-Marseille Université, 2013.

* Riikka Huusari, Hachem Kadri and Cécile Capponi,
  "Multi-View Metric Learning in Vector-Valued Kernel Spaces"
  in International Conference on Artificial Intelligence and Statistics (AISTATS) 2018

Copyright
~~~~~~~~~

Université d'Aix Marseille (AMU) -
Centre National de la Recherche Scientifique (CNRS) -
Université de Toulon (UTLN).

Copyright © 2017-2018 AMU, CNRS, UTLN

License
~~~~~~~

**scikit-multimodallearn** is free software: you can redistribute it and/or modify
it under the terms of the **New BSD License**
