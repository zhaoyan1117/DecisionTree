from __future__ import absolute_import
from __future__ import division

import numpy as np
from scipy.stats import mode

from ._node import Node
from .util import iterator_with_progress

class DecisionTree:

    def __init__(self, impurity, segmentor, **kwargs):
        self._impurity = impurity
        self._segmentor = segmentor
        self._max_depth = kwargs.get('max_depth', None)
        self._min_points = kwargs.get('min_points', None)
        self._root = None

    def train(self, data, labels):
        self._root = self._generate_node(data, labels, 0)

    def predict(self, data):
        if not self._root:
            raise StandardError("Decision tree has not been trained.")
        size = data.shape[0]
        predictions = np.empty((size,))
        for i in iterator_with_progress(size):
            predictions[i] = self._predict_single(data[i])
        return predictions

    def score(self, data, labels):
        if not self._root:
            raise StandardError("Decision tree has not been trained.")
        predictions = self.predict(data)
        correct_count = np.count_nonzero(predictions == labels)
        return round(correct_count / labels.shape[0], 2)

    def _predict_single(self, datum):
        cur_node = self._root
        while not cur_node.is_leaf:
            cur_node = cur_node.get_child(datum)
        return cur_node.label

    def _generate_node(self, data, labels, cur_depth):
        if self._terminate(data, labels, cur_depth):
            return self._generate_leaf_node(cur_depth, labels)
        else:
            sr, left_indices, right_indices = self._segmentor(data, labels, self._impurity)

            if not sr:
                return self._generate_leaf_node(cur_depth, labels)

            left_data, left_labels = data[left_indices], labels[left_indices]
            right_data, right_labels = data[right_indices], labels[right_indices]

            return Node(cur_depth, mode(labels)[0][0],
                        split_rules=sr,
                        left_child=self._generate_node(left_data, left_labels, cur_depth+1),
                        right_child=self._generate_node(right_data, right_labels, cur_depth+1),
                        is_leaf=False)

    def _generate_leaf_node(self, cur_depth, labels):
        return Node(cur_depth, mode(labels)[0][0], is_leaf=True)

    def _terminate(self, data, labels, cur_depth):
        if self._max_depth != None and cur_depth == self._max_depth:
            # max depth reached
            return True
        elif self._min_points != None and labels.size < self._min_points:
            # min depth reached
            return True
        elif np.unique(labels).size == 1:
            return True
        else:
            return False
