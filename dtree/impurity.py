from __future__ import absolute_import
from __future__ import division

import numpy as np
from math import log

def entropy(left_label_hist, right_label_hist):
    left_total = np.sum(left_label_hist)
    right_total = np.sum(right_label_hist)
    total = left_total + right_total

    left_entropy = 0.0
    for count in left_label_hist:
        freq = count/left_total
        left_entropy -= freq * log(freq, 2)

    right_entropy = 0.0
    for count in right_label_hist:
        freq = count/right_total
        right_entropy -= freq * log(freq, 2)

    return (left_total/total) * left_entropy + (right_total/total) * right_entropy
