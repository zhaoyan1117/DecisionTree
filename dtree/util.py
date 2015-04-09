from __future__ import absolute_import
from __future__ import division

from sys import stdout
from scipy.stats import itemfreq
import numpy as np
import math, operator

def iterate_with_progress(collections):
    cursor = '.'
    last_percent = -1
    length = len(collections)

    for index, item in enumerate(collections):
        cur_percent = int(100.0 * ((index+1) / length))
        if cur_percent > last_percent:
            last_percent = cur_percent
            stdout.write('\r' + cursor * cur_percent + " %d%%" % cur_percent)
            if cur_percent == 100:
                stdout.write('\n')
            stdout.flush()
        yield item

def get_labels_freq(labels, normalize=True):
    freq = itemfreq(labels)
    labels_freq = dict({label : count for label, count in freq})
    if normalize:
        normalize_values(labels_freq)
    return labels_freq

# http://stackoverflow.com/questions/16417916/normalizing-dictionary-values
def normalize_values(d):
    factor = 1.0 / math.fsum(d.itervalues())
    for k in d:
        d[k] = d[k]*factor
    key_for_max = max(d.iteritems(), key=operator.itemgetter(1))[0]
    diff = 1.0 - math.fsum(d.itervalues())
    d[key_for_max] += diff
    return d
