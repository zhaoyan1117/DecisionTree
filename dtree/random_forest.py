from __future__ import absolute_import
from __future__ import division

import numpy as np
from scipy.stats import mode
from .decision_tree import DecisionTree
from .util import iterate_with_progress

class RandomForest:

    def __init__(self, impurity, segmentor, **kwargs):
        self._impurity = impurity
        self._segmentor = segmentor
        self._boost_p = kwargs.get('boost_p', 0.5)
        assert 0.0 < self._boost_p and self._boost_p <= 1.0
        self._num_trees = kwargs.get('num_trees', 10)
        assert self._num_trees > 0
        self._max_depth = kwargs.get('max_depth', None)
        self._min_samples = kwargs.get('min_samples', 2)
        self._trees = []

    def train(self, data, labels):
        self._klasses = np.unique(labels)
        for _ in iterate_with_progress(xrange(self._num_trees)):
            sampled_data, sampled_labels = self._sample_data_labels(data, labels)
            tree = DecisionTree(self._impurity,
                                self._segmentor,
                                max_depth=self._max_depth,
                                min_samples=self._min_samples)
            tree.train(sampled_data, sampled_labels)
            self._trees.append(tree)

    def predict(self, data):
        if not self._trees:
            raise StandardError("Random forest has not been trained.")
        def draw_votes(probs):
            avg_probs = {}
            for klass in self._klasses:
                total_prob = sum([prob.get(klass, 0.0) for prob in probs])
                avg_probs[klass] = total_prob / self._num_trees
            return max(avg_probs, key=lambda klass : avg_probs[klass])
        tree_results = np.array([tree.predict(data, True) for tree in self._trees])
        return np.apply_along_axis(draw_votes, 0, tree_results)

    def score(self, data, labels):
        if not self._trees:
            raise StandardError("Random forest has not been trained.")
        predictions = self.predict(data)
        correct_count = np.count_nonzero(predictions == labels)
        return correct_count / labels.shape[0]

    def prune(self, data, labels):
        for tree in iterate_with_progress(self._trees):
            tree.prune(data, labels)
        return self.score(data, labels)

    def _sample_data_labels(self, data, labels):
        if self._boost_p == 1.0:
            return data, labels

        num_data = data.shape[0]
        assert num_data == len(labels)
        data_indices = np.random.permutation(num_data)
        num_data_sampled = int(num_data * self._boost_p)
        sampled_data = data[data_indices[:num_data_sampled],:]
        sampled_labels = labels[data_indices[:num_data_sampled]]

        return sampled_data, sampled_labels
