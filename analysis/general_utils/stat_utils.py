import scipy
import scipy.stats
import numpy as np
from scipy.stats import pearsonr

def mean_confidence_interval(data, confidence=0.95):
    '''
    Return confidence interval from list.

    Args:
        data (np.ndarray or list) : the list
        confidence (float) : the confidence interval to calculate 0.95 for 95%
    '''
    data = np.array(data)
    data = data[~np.isnan(data)]
    a = 1.0 * data
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def get_pearsonr(a, b):
    '''
    Return Pearson correlation coefficient given 2 arrays
    '''
    return pearsonr(a, b)
