from __future__ import absolute_import

import numpy as np

class Node:

    def __init__(self, depth, label_prob, **kwargs):
        self.depth = depth
        self.label_prob = label_prob
        self.is_leaf = kwargs.get('is_leaf', False)
        self._split_rules = kwargs.get('split_rules', None)
        self._left_child = kwargs.get('left_child', None)
        self._right_child = kwargs.get('right_child', None)

        if not self.is_leaf:
            assert self._split_rules
            assert self._left_child
            assert self._right_child

    def get_child(self, datum):
        if self.is_leaf:
            raise StandardError("Leaf node does not have children.")
        feature_index, threshhold = self.split_rules
        if datum[feature_index] < threshhold:
            return self.left_child
        else:
            return self.right_child

    def get_label_prob(self, label):
        return self.label_prob.get(label, 0.0)

    @property
    def label(self):
        if not hasattr(self, '_label'):
            self._label = max(self.label_prob,
                                        key=lambda label : self.label_prob[label])
        return self._label

    @property
    def split_rules(self):
        if self.is_leaf:
            raise StandardError("Leaf node does not have split rule.")
        return self._split_rules

    @property
    def left_child(self):
        if self.is_leaf:
            raise StandardError("Leaf node does not have split rule.")
        return self._left_child

    @property
    def right_child(self):
        if self.is_leaf:
            raise StandardError("Leaf node does not have split rule.")
        return self._right_child
