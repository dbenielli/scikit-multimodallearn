# -*- coding: utf-8 -*-
# ######### COPYRIGHT #########
#
# Copyright(c) 2020
# -----------------
#
# * Université d'Aix Marseille (AMU) -
# * Centre National de la Recherche Scientifique (CNRS) -
# * Université de Toulon (UTLN).
# * Copyright © 2019-2020 AMU, CNRS, UTLN
#
# Contributors:
# ------------
#
# * Sokol Koço <sokol.koco_AT_lis-lab.fr>
# * Cécile Capponi <cecile.capponi_AT_univ-amu.fr>
# * Florent Jaillet <florent.jaillet_AT_math.cnrs.fr>
# * Dominique Benielli <dominique.benielli_AT_univ-amu.fr>
# * Riikka Huusari <rikka.huusari_AT_univ-amu.fr>
# * Baptiste Bauvin <baptiste.bauvin_AT_univ-amu.fr>
# * Hachem Kadri <hachem.kadri_AT_lis-lab.fr>
#
# Description:
# -----------
#
# The multimodal package implement classifiers multiview, 
# MumboClassifier class, MuCumboClassifier class, MVML class, MKL class.
# compatible with sklearn
#
# Version:
# -------
#
# * multimodal version = 0.0.dev0
#
# Licence:
# -------
#
# License: New BSD License
#
#
# ######### COPYRIGHT #########
#
#
#
#
#


from __future__ import print_function, division

import os


def get_dataset_path(filename):
    """Return the absolute path of a reference dataset for tests

    - Input parameter:

    :param str filename: File name of the file containing reference data
        for tests (which must be in ``skgilearn/tests/datasets/``)

    - Output parameters:

    :returns: The absolute path where the file with name **filename** is stored
    :rtype: str

    """

    datasets_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(datasets_path, filename)
