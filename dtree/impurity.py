from __future__ import absolute_import
from __future__ import division

import numpy as np
from math import log

class Entropy:

    def __call__(self, left_label_hist, right_label_hist):
        left_bincount, right_bincount = left_label_hist[:,1], right_label_hist[:,1]
        left_total, right_total = np.sum(left_bincount), np.sum(right_bincount)

        left_entropy = self._cal_entropy(left_bincount, left_total)
        right_entropy = self._cal_entropy(right_bincount, right_total)

        total = left_total + right_total

        return (left_total/total) * left_entropy + (right_total/total) * right_entropy

    def _cal_entropy(self, bincount, total):
        entropy = 0.0
        for count in bincount:
            freq = count/total
            entropy -= freq * log(freq, 2)
        return entropy
