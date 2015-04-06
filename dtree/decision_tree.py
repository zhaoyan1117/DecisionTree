from __future__ import absolute_import

import numpy as np
from scipy.stats import mode

from _leaf_node import LeafNode
from _node import Node
from util import iterator_with_progress

class DecisionTree:

    def __init__(self, impurity, segmentor, **params):
        self._impurity = impurity
        self._segmentor = segmentor
        self._max_depth = params.get('max_depth', None)
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

    def _predict_single(self, datum):
        cur_node = self._root
        while type(cur_node) != LeafNode:
            cur_node = cur_node.get_child(datum)
        return cur_node.label

    def _generate_node(self, data, labels, cur_depth):
        if self._max_depth != None and cur_depth == self._max_depth:
            return self._generate_leaf_node(labels)
        else:
            sr, left_indices, right_indices = self._segmentor(data, labels, self._impurity)

            left_data, left_labels = data[left_indices], labels[left_indices]
            right_data, right_labels = data[right_indices], data[right_indices]

            return Node(sr, cur_depth + 1,
                        self._generate_node(left_data, left_labels, next_depth),
                        self._generate_node(right_data, right_labels, next_depth))

    def _generate_leaf_node(self, labels):
        return LeafNode(mode(labels)[0][0])
