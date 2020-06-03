import math
import numpy as np
def truncate(number, digits) -> float:
    '''
    Truncate number to n nearest digits. Set to -ve for decimal places

    Args:
        number (float) : the number to truncate
        digits (int) : nearest digits. 0 truncate to 1. 1 truncate to 10. -1
                                        truncate to 0.1
    '''
    stepper = pow(10.0, digits)
    return math.trunc(stepper * number) / stepper

def merge_l_l(l_l, downsample_length):
    '''
    Merge list of lists over downsample length [[..], [.], [..] ] -> [[...], [....]]
    '''
    cut_end_length = (len(l_l) % downsample_length)

    reshaped_l_l = np.array(l_l[0:-cut_end_length]).reshape(-1, downsample_length)
    downsampled_l_l = []
    flatten = lambda l: [item for sublist in l for item in sublist]

    for i in range(reshaped_l_l.shape[0]):
        downsampled_l_l.append(flatten(reshaped_l_l[i, :]))
    return downsampled_l_l
