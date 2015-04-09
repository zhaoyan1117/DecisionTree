from __future__ import absolute_import
from __future__ import division

from multiprocessing import cpu_count, Pool
import time, signal

import numpy as np
import random, math

from .decision_tree import DecisionTree
from .util import iterate_with_progress, normalize_values

#################################
# Multi-process funcs & klasses #
#################################
class KeyboardInterruptError(Exception): pass

def prune_tree(args):
    try:
        tree, data, labels = args
        tree.prune(data, labels)
        return tree
    except KeyboardInterrupt:
        raise KeyboardInterruptError()

class AdaBoost:

    def __init__(self, impurity, segmentor, **kwargs):
        self._impurity = impurity
        self._segmentor = segmentor
        self._num_trees = kwargs.get('num_trees', 10)
        assert self._num_trees > 0
        self._max_depth = kwargs.get('max_depth', None)
        self._min_samples = kwargs.get('min_samples', 2)
        self._trees = []
        self._alphas = []

    def train(self, data, labels):
        assert len(data) == len(labels)
        self._klasses = np.unique(labels)

        distributions = normalize_values({i:1 for i in xrange(len(data))})
        for _ in iterate_with_progress(xrange(self._num_trees)):
            sampled_data, sampled_labels = self._sample_data_labels(data, labels, distributions)
            tree = DecisionTree(self._impurity,
                                self._segmentor,
                                max_depth=self._max_depth,
                                min_samples=self._min_samples)

            tree.train(sampled_data, sampled_labels)
            predictions = tree.predict(data)

            error = sum([distributions[i] for i in np.nonzero(predictions != labels)[0]])
            alpha = float('inf') if error == 0.0 else 0.5 * math.log((1.0 - error)/error)

            self._trees.append(tree)
            self._alphas.append(alpha)

            for i in xrange(len(data)):
                weight = alpha if predictions[i] != labels[i] else -alpha
                distributions[i] *= math.e ** weight
            distributions = normalize_values(distributions)
        self._alphas = np.array(self._alphas)

    def predict(self, data):
        if not self._trees:
            raise StandardError("AdaBoost has not been trained.")
        def weight(results):
            results[np.nonzero(results == 0)[0]] = -1
            strong_result = np.dot(self._alphas, results)
            return int(strong_result >= 0)
        tree_results = np.array([tree.predict(data) for tree in self._trees])
        return np.apply_along_axis(weight, 0, tree_results)

    def score(self, data, labels):
        if not self._trees:
            raise StandardError("AdaBoost has not been trained.")
        predictions = self.predict(data)
        correct_count = np.count_nonzero(predictions == labels)
        return correct_count / len(labels)

    def prune(self, data, labels):
        args_list = []
        for tree in self._trees:
            args_list.append([tree, data, labels])

        num_processes = cpu_count()
        print 'Prune in parallel with {0} processes.'.format(num_processes)
        pool = Pool(num_processes)
        try:
            start = time.time()
            self._trees = pool.map(prune_tree, args_list)
            print 'Pruning takes {0} seconds.'.format(int(time.time() - start))
            pool.close()
            return self.score(data, labels)
        except KeyboardInterrupt:
            pool.terminate()
        except Exception, e:
            pool.terminate()
        finally:
            pool.join()

    def _sample_data_labels(self, data, labels, distributions):
        if not sum(distributions.itervalues()) == 1.0:
            distributions = normalize_values(distributions)
        random.seed()
        n, m = data.shape
        sampled_data = np.empty((0,data.shape[1]), dtype=data.dtype)
        sampled_labels = np.empty((0,), dtype=labels.dtype)
        draws = sorted([random.random() for _ in xrange(n)])
        sample_i, data_i, cdf = 0, 0, distributions[0]
        while sample_i < n:
            if draws[sample_i] < cdf:
                sample_i += 1
                sampled_data = np.append(sampled_data, data[data_i].reshape(1,m), axis=0)
                sampled_labels = np.append(sampled_labels, labels[data_i])
            else:
                data_i += 1
                cdf += distributions[data_i]
        return sampled_data, sampled_labels
