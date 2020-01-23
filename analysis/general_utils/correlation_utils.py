import numpy as np
import scipy.signal as ss
from analysis.general_utils import compare_astro_utils
from preprocessing import analysis_pp
import skimage
import scipy

def get_max_cross_correlation(signal_a, signal_b, normalize=False):
    '''
    Signal 1D cross correlation, between two signals returning index of maximum
    correlation and its corresponding value.
    Signal a size should be equal or larger than signal b size
    '''
    a = signal_a
    b = signal_b

    if np.std(a) == 0:
        return 0, 0
    if np.std(b) == 0:
        return 0, 0

    if normalize:
        a = (a - np.mean(a)) / (np.std(a))
        b = (b - np.mean(b)) / (np.std(b) * len(b))

    corr = np.correlate(a, b)
    max_corr_i = np.argmax(corr)
    max_corr = corr[max_corr_i]
    return max_corr, max_corr_i

def get_cross_correlation_2D_info_compare(grid_1, grid_2, normalize=True, mode='same'):
    '''
    Image cross correlation between two grids

    Args:
        grid_1 : grid 1
        grid_2 : grid 2
        normalize : True for pearson correlation
        mode : 'scipy.signal.correlate' mode
    Returns:
        corr_res : corr_res_sample
        max_corr_sample : max correlation value
        move_vector : move vector
        (x,y) : (y_coord, x_coord) of max correlation

    '''
    if (grid_2.shape[0] > grid_1.shape[0]) or (grid_2.shape[1] > grid_1.shape[1]):
        print('Grid 2 must have less or equal dimensions than Grid 1')
        return

    if normalize:
        grid_1_norm = (grid_1 - np.mean(grid_1)) / np.std(grid_1)
        grid_2_norm = (grid_2 - np.mean(grid_2)) / np.std(grid_2)

        corr_res = ss.correlate2d(grid_1_norm, grid_2_norm, mode=mode, boundary='fill', fillvalue=0)
        corr_res = corr_res / (grid_2_norm.shape[0]*grid_2_norm.shape[1])
    else:
        corr_res = ss.correlate2d(grid_1, grid_2, mode=mode, boundary='fill', fillvalue=0)

    y_coord, x_coord = np.unravel_index(corr_res.argmax(), corr_res.shape)

    center_y = int(np.ceil(corr_res.shape[0] / 2) - 1)
    center_x = int(np.ceil(corr_res.shape[1] / 2) - 1)

    move_vector = [(y_coord - center_y), (x_coord-center_x)]
    print('Move vector:', move_vector)
    print('Max corr:', corr_res[y_coord,x_coord])
    return corr_res, corr_res[y_coord, x_coord], move_vector, (y_coord, x_coord)

def get_corr_astro_samples_v2(astro_xc, astro_base, grid_l_pair=None, p=0.05, n_samples=1, fill_type='dt'):
    '''
    Deprecated

    Get correlations between fake samples
    '''
    astro_l_pair = [astro_xc, astro_base]
    #astro_l_pair contains [astro_1, astro_2], astro_1 is the astrocyte we will apply cross correlation with on the samples
    #astro_2 is the astrocyte we try to copy from

    if grid_l_pair is None:
        astro_filt_l, astro_all_filt, astro_nz_bool_l, astro_all_nz_bool = compare_astro_utils.get_filters_compare(astro_l_pair, p=p)
    else:
        astro_filt_l, astro_all_filt, astro_nz_bool_l, astro_all_nz_bool = compare_astro_utils.get_filters_compare_from_grids(grid_l_pair, p=p)

    sample_l = []
    max_corr_l = []
    corr_res_l = []

    for sample_i in range(n_samples):
        print('Sample {}'.format(sample_i))
        sample = compare_astro_utils.get_fake_astrocyte_sample_v2(astro_l_pair[1], astro_filt_l[1])

        if fill_type == 'mean':
            #Fill the sample with mean value of astro_filt_l
            mean_filt_value = np.mean(astro_filt_l[1][astro_filt_l[1] > 0])
            sample[sample == 1] = mean_filt_value
        if fill_type == 'dt':
            #Fill value with distance from nearest empty space. This increases value to center
            sample = scipy.ndimage.distance_transform_edt(sample)

            m1 = np.mean(sample)
            s1 = np.std(sample)

            m2 = np.mean(astro_filt_l[1])
            s2 = np.std(astro_filt_l[1])

            sample = m2 + (sample - m1)*(s2/s1)

        corr_res_sample, max_corr_sample, move_vector_sample, max_coord_sample = get_cross_correlation_2D_info_compare(astro_filt_l[0], sample)
        max_corr_l.append(max_corr_sample)
        sample_l.append(sample)
        corr_res_l.append(corr_res_sample)

    return {'max_corr_l' : max_corr_l,
            'sample_l' : sample_l,
            'corr_res_l' : corr_res_l}

def get_splits_corr(astroA, num_frames_splits_l=[3000, 6000, 12000, 24000], p=0.05, max_comparisons=50):
    '''
    Args:
        Apply frame splits and generate correlations between frame split grids
    '''
    #~5 minutes, ~10 minutes, ~20 minutes, ~40 minute
    split_corrs_d = {}

    for split_frames in num_frames_splits_l:
        print('Split frames: {}'.format(split_frames))
        event_grid_splits_l = compare_astro_utils.split_astro_grid(astroA, split_frames=split_frames, bk='default')

        split_corrs_d[str(split_frames)] = {'corr_res_l' : [],
                                         'max_corr_l' : [],
                                         'move_vector_l' : [],
                                         'event_grid_splits_l' : event_grid_splits_l}

        pairs = [(i, j ) for i in range(len(event_grid_splits_l)) for j in range(i+1, len(event_grid_splits_l))]

        if len(pairs) > max_comparisons :
            print('Max comparisons > len pairs, {} > {}'.format(max_comparisons, len(pairs)))
            pairs_perm = np.random.permutation(pairs)
            pairs = pairs_perm[:max_comparisons]

        for i, j in pairs:
            print(i, j)
            grid_l_pair = [event_grid_splits_l[i], event_grid_splits_l[j]]
            astro_filt_l_tmp, astro_all_filt_tmp, astro_nz_bool_l_tmp, astro_all_nz_bool_tmp = compare_astro_utils.get_filters_compare_from_grids(grid_l_pair, p=p)
            corr_res, max_corr, move_vector, max_coord = get_cross_correlation_2D_info_compare(astro_filt_l_tmp[0], astro_filt_l_tmp[1])

            split_corrs_d[str(split_frames)]['corr_res_l'].append(corr_res)
            split_corrs_d[str(split_frames)]['max_corr_l'].append(max_corr)
            split_corrs_d[str(split_frames)]['move_vector_l'].append(move_vector)

            print(i, j, max_corr)
    return split_corrs_d

def normalize_grid(grid, border=None):
    '''
    Standardize grid

    Args:
        grid (np.ndarray): The 2D grid
        border (np.ndarray) : Optional, boolean 2D grid corresponding to which values to apply standardization
    Returns:
        Standardized grid
    '''
    grid = np.copy(grid)
    if border is None:
        grid = (grid - np.mean(grid)) / np.std(grid)
    else:
        grid[border == 0] = np.nan
        nan_mean = np.nanmean(grid)
        nan_std = np.nanstd(grid)

        grid[~np.isnan(grid)] -= nan_mean
        grid[~np.isnan(grid)] /= nan_std
    grid[np.isnan(grid)] = 0

    if border is None:
        return grid
    else:
        return grid

"""
####################################
UNUSED FUNCTIONS
####################################
"""

def myFWHM(X, index_top, frac=2):
    max_val = X[index_top]
    #Keep going one direction until we stop dropping and apply same to other direction
    left_min_i, right_min_i = get_pyramid_i(X, index_top)
    min_val = min(X[left_min_i], X[right_min_i])
    mid_val = (max_val + min_val) / 2

    #Find first left and last right index larger than mid_val
    X_slice = X[left_min_i:right_min_i+1]
    greater_indexes = np.where((X_slice - mid_val) >= 0)[0]
    first_left, last_right = greater_indexes[0], greater_indexes[-1]

    return abs(last_right - first_left)

def get_pyramid_i(X, index_top):
    '''
    Given 1D array of values in X and index_top representing the starting point
    Move left and right (individually) until we stop getting decreasing values
    Return the final indices we get
    '''
    #rightmost is our top_i
    left_slice = X[:index_top+1]
    left_diff = left_slice[:-1] - left_slice[1:]
    #-ve values imply decreasing values from right to left
    left_i = np.where(left_diff >= 0)[0]
    if len(left_i) == 0:
        left_min_i = 0
    else:
        left_min_i = left_i[-1] + 1
    #leftmost is our top_i
    right_slice = X[index_top:]
    #-ve values imply increasing values from left to right
    right_diff = right_slice[1:] - right_slice[:-1]
    right_i = np.where(right_diff >= 0)[0]
    if len(right_i) == 0:
        right_min_i = len(X) - 1
    else:
        right_min_i = right_i[0] + index_top
    return left_min_i, right_min_i

def filter_non_peaks(arr, indices):
    new_indices = []
    removed_non_peaks = 0
    for i in indices:
        if i < 1 or i >= len(arr):
            print('Out of bounds indice', i)

        #FILTER OUT ANY INDICES NOT AT PEAK
        l, r = get_pyramid_i(arr[i-1:i+2], index_top=1)
        # If both sides increasing values (r-l) = 0, if one side (r-l) = 1
        if (r - l) < 2:
            #print('increasing at one side:', roi_dict_x['X1'][corr_indices[i]-1:corr_indices[i]+2], i)
            removed_non_peaks += 1
            continue
        else:
            new_indices.append(i)
    print('Filtered out: {} non-peaks'.format(removed_non_peaks))
    return new_indices
