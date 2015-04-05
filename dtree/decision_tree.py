from __future__ import absolute_import

import numpy as np

class DecisionTree:

    def __init__(self, impurity, segmentor, **params):
        self.impurity = impurity
        self.segmentor = segmentor
        self.depth = params.get('depth', None)
        self.root = None
