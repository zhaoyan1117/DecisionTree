from __future__ import absolute_import
from __future__ import division

import numpy as np
from scipy.stats import mode
from decision_tree import DecisionTree

class RandomForest:
    def __init__(self, impurity, segmentor, **kwargs):
        self._impurity = impurity
        self._segmentor = segmentor
