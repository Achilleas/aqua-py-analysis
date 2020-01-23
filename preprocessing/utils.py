import numpy as np
import pandas

def get_voltage_increase(binary_phase):
    '''
    Used to obtain voltage increase
    Args:
        binary_phase (np.ndarray): 1D array of binary values
    Returns:
        voltage_incrase (np.ndarray): 1 where 0->1, 0 otherwize
    '''
    voltage_increase = binary_phase[1:] - binary_phase[:-1]
    #Only want changes from 0 -> 1 not from 1 -> 0 to be set to 1
    voltage_increase[np.where(voltage_increase == -1)] = 0
    return voltage_increase

def hampel(vals_orig, k=7, num_std=3, only_indices=False):
    '''
    Args:

    Returns:
    vals: pandas series of values from which to remove outliers
    k: size of window (including the sample; 7 is equal to 3 on either side of value)
    '''

    #Make copy so original not edited
    vals = vals_orig.copy()

    #Hampel Filter
    L = 1.4826
    rolling_median = vals.rolling(window=k, center=True).median()
    MAD = lambda x: np.median(np.abs(x - np.median(x)))
    rolling_MAD = vals.rolling(window=k, center=True).apply(MAD, raw=True)
    threshold = num_std * L * rolling_MAD
    difference = np.abs(vals - rolling_median)
    outlier_idx = (difference > threshold)

    if only_indices:
        return np.where(outlier_idx)[0]
    else:
        vals[outlier_idx] = rolling_median[outlier_idx]
        return vals
