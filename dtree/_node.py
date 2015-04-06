from __future__ import absolute_import

import numpy as np

class Node:

    def __init__(self, sr, d, lc, rc):
        self.split_rules = sr
        self.depth = d
        self.left_child = lc
        self.right_child = rc

    def get_child(self, datum):
        feature_index, threshhold = sr
        if datum[feature_index] < threshhold:
            return self.left_child
        else:
            return self.right_child
