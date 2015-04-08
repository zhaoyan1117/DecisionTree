from __future__ import absolute_import
from __future__ import division

import numpy as np
from scipy.stats import mode

from .util import get_labels_freq
from ._node import Node

class DecisionTree:

    def __init__(self, impurity, segmentor, **kwargs):
        self._impurity = impurity
        self._segmentor = segmentor
        self._max_depth = kwargs.get('max_depth', None)
        self._min_samples = kwargs.get('min_samples', 2)
        self._root = None
        self._nodes = []
        self._pruned_nodes = []

    def train(self, data, labels):
        self._root = self._generate_node(data, labels, 0)

    def predict(self, data, distribution=False):
        if not self._root:
            raise StandardError("Decision tree has not been trained.")
        size = data.shape[0]
        return_type = np.object if distribution else np.float64
        predictions = np.empty((size,), dtype=return_type)
        for i in xrange(size):
            predictions[i] = self._predict_single(data[i], distribution)
        return predictions

    def score(self, data, labels):
        if not self._root:
            raise StandardError("Decision tree has not been trained.")
        predictions = self.predict(data)
        correct_count = np.count_nonzero(predictions == labels)
        return correct_count / labels.shape[0]

    def prune(self, data, labels):
        while True:
            best_score = self.score(data, labels)
            pruned_node = None
            for node in self._nodes:
                if not node.is_leaf:
                    node.is_leaf = True
                    cur_score = self.score(data, labels)
                    if cur_score > best_score:
                        best_score = cur_score
                        pruned_node = node
                    node.is_leaf = False
            if pruned_node:
                self._pruned_nodes.append(pruned_node)
                pruned_node.is_leaf = True
            else:
                return best_score, len(self._pruned_nodes)

    def _predict_single(self, datum, distribution):
        cur_node = self._root
        while not cur_node.is_leaf:
            cur_node = cur_node.get_child(datum)
        if distribution:
            return cur_node.label_prob
        else:
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

            node = Node(cur_depth, get_labels_freq(labels),
                        split_rules=sr,
                        left_child=self._generate_node(left_data, left_labels, cur_depth+1),
                        right_child=self._generate_node(right_data, right_labels, cur_depth+1),
                        is_leaf=False)
            self._nodes.append(node)
            return node

    def _generate_leaf_node(self, cur_depth, labels):
        node = Node(cur_depth, get_labels_freq(labels), is_leaf=True)
        self._nodes.append(node)
        return node

    def _terminate(self, data, labels, cur_depth):
        if self._max_depth != None and cur_depth == self._max_depth:
            # maximum depth reached.
            return True
        elif labels.size < self._min_samples:
            # minimum number of samples reached.
            return True
        elif np.unique(labels).size == 1:
            return True
        else:
            return False
