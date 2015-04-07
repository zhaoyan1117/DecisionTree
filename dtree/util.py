from __future__ import absolute_import
from __future__ import division

from sys import stdout
from scipy.stats import itemfreq
import numpy as np

def iterator_with_progress(n):
    cursor = '.'
    last_percent = -1
    for i in xrange(n):
        cur_percent = int(100.0 * (i+1) / n)
        if cur_percent > last_percent:
            last_percent = cur_percent
            stdout.write('\r' + cursor * cur_percent + " %d%%" % cur_percent)
            if cur_percent == 100:
                stdout.write('\n')
            stdout.flush()
        yield i

def get_labels_freq(labels, normalize=True):
    freq = itemfreq(labels)
    if normalize:
        total = np.sum(freq[:,1])
        return dict({label : count/total for label, count in freq})
    else:
        return dict({label : count for label, count in freq})
