from analysis.general_utils import saving_utils
import os,sys,glob
import numpy as np
import scipy, scipy.signal
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import savgol_filter

'''
########################################################################
#############################DEPRECATED#################################
########################################################################
'''

def get_grid_signal(grid_volume, noise_check_ratio=3, binary_signals=False):
    grid_volume = np.copy(grid_volume)
    #Determine noise boxes of grid volume based on its statistics
    mean_grid, std_grid, min_grid, max_grid = get_grid_statistics(grid_volume)

    noise_check_grid = get_noise_check_grid(mean_grid, std_grid, min_grid, max_grid, noise_check_ratio=noise_check_ratio)
    #Set grid volume noise boxes to 0

    if binary_signals:
        #Set noise boxes to value 1
        grid_volume = set_noise(grid_volume, noise_check_grid, set_value=0)
        #Just set signal boxes to value 1
        result_volume = set_signals(grid_volume, noise_check_grid, set_value=1)
    else:
        #Set noise boxes to non-zero minimum of volume
        grid_volume = set_noise(grid_volume, noise_check_grid, set_value=0)
        #Apply df/f to grid volume on each box
        result_volume =  apply_grid_df(grid_volume)
    #Sum up the results of each box over the z axis
    result_sum_volume = np.sum(result_volume, axis=(0)) / (result_volume.shape[0])
    return result_sum_volume

def read_grid_folder(tif_folder, box_size=5):
    '''
    Read folder of tifs, create grid for each tif and concatenate as singular volume
    '''
    grid_volume = None
    tif_paths = glob.glob(tif_folder + '/*')
    for i, tif_path in enumerate(tif_paths):
        print('{}/{} {}'.format(i+1, len(tif_paths), tif_path))
        grid_volume_i = setup_grid_from_file(tif_path=tif_path, box_size=box_size)
        if i == 0:
            grid_volume = grid_volume_i
        else:
            grid_volume = np.concatenate([grid_volume, grid_volume_i], axis=0)
    return grid_volume

def setup_grid_from_file(tif_path, box_size=5):
    '''
    Create grid structure and average boxes given box size
    '''
    source_vol = saving_utils.read_tif_volume(tif_path)
    #swap axes as dim 1 is y and dim 2 is x coord:
    source_vol = np.swapaxes(source_vol, 1, 2)
    #Create grid structure and take average over each box specified by box_size
    cut_source_vol = make_box_divisible(source_vol, box_size)
    #(nstacks, 512, 512) -> (nstacks, 102, 102, 5,5)
    box_source_vol = blockshaped_vol(cut_source_vol, box_size)
    #(nstacks, 102,102,5,5) -> (nstacks, 102, 102)
    avg_boxes_source_vol = np.mean(box_source_vol, axis=(3,4))
    return avg_boxes_source_vol

def get_grid_statistics(grid_volume):
    mean_grid_vol = np.mean(grid_volume, axis=(0))
    std_grid_vol = np.std(grid_volume, axis=(0))
    min_grid_vol = np.min(grid_volume, axis=(0))
    max_grid_vol = np.max(grid_volume, axis=(0))

    return mean_grid_vol, std_grid_vol, min_grid_vol, max_grid_vol

def get_noise_check_grid(mean_grid, std_grid, min_grid, max_grid, noise_check_ratio=3):
    '''
    Given (102, 102) statistics, determine if box at each location is noise
    Returns (102, 102) binary values if each box is noise
    '''
    #Find boxes that contain signals
    #   A box contains a signal if it contains at least one box over z axis
    #   with value > mean + 3*std of stack
    noise_check_grid = ((np.abs(max_grid - mean_grid)) /
                    (np.abs(min_grid - mean_grid))) < noise_check_ratio
    return noise_check_grid

def set_noise(grid_volume, noise_check_grid, set_value=0):
    grid_noise_locs_x, grid_noise_locs_y = np.nonzero(noise_check_grid)
    #print('Grid noise locs:', len(grid_noise_locs_x))

    grid_signal_locs_x, grid_signal_locs_y = np.nonzero(np.invert(noise_check_grid))
    #print('Grid signal locs', len(grid_signal_locs_x))

    grid_volume[:, grid_noise_locs_x, grid_noise_locs_y] = set_value
    return grid_volume

def set_signals(grid_volume, noise_check_grid, set_value=0):
    grid_signal_locs_x, grid_signal_locs_y = np.nonzero(np.invert(noise_check_grid))
    grid_volume[:, grid_signal_locs_x, grid_signal_locs_y] = set_value
    return grid_volume

def apply_grid_df(grid_volume):
    f_0 = grid_volume.min(axis=(0))
    epsilon = 1e-20
    df_volume = (grid_volume - f_0) / (f_0 + epsilon)
    return df_volume

def make_box_divisible(volume, box_size=5):
    new_shape = [volume.shape[0],
                volume.shape[1] - (volume.shape[1] % box_size),
                volume.shape[2] - (volume.shape[2] % box_size)]
    return volume[0:new_shape[0], 0:new_shape[1], 0:new_shape[2]]

def blockshaped_vol(arr, box_size):
    '''
    Given arr (n, h, w) and box_size
    return array (x_n, y_n, box_size, box_size) such that
    (x_i, y_i) represents the grid coordinate of a box_size x box_size box.
    '''
    n, h, w = arr.shape

    #Reshape to (n, h // box_size, box_size, w // box_size, box_size)
    #Swap axs to (n, h // box_size, w // box_size, box_size, box_size)
    return (arr.reshape(n, h//box_size, box_size, -1, box_size)
               .swapaxes(2,3))


def find_signal_peaks(grid_volume, noise_check_ratio, continuous_threshold=None):
    '''
    Takes in a grid volume and finds signal peaks as boolean matrix
    Params:
        grid_volume:        the grid volume
        box_size:           the box size
        noise_check_ratio : how many times abs(curr_val - mean) must be
                            compared to abs(mean-min) to consider as signal
        filter_continuous : default None. If set to value filters out
                            continuous signals from signal starts. e.g
                            1 0 1 0 1 0 0 1 0 0 0 1 0 0 1 0 1
                            with filter_continuous = 2
                            1 0 0 0 0 0 0 1 0 0 0 1 0 0 1 0

    - For each box from mask of signal boxes
    - Determine number of unique signals in each box
        - unique signals must be above specific ratio std
        - unique signals must be peaks
        - unique signals must be certain distance d between other unique signals
    - Return result as volume
    '''
    grid_volume = np.copy(grid_volume)
    #print('Obtaining statistics...')
    mean_grid_vol, std_grid_vol, min_grid_vol, max_grid_vol = get_grid_statistics(grid_volume)
    #print('Done')
    peaks_volume = np.zeros(grid_volume.shape)
    #Find signals: must be peaks and have minimum distance between each other and satisfy noise_check ratio
    for i in range(grid_volume.shape[1]):
        for j in range(grid_volume.shape[2]):
            #print('{}/{}'.format(i*grid_volume.shape[1] + j + 1, grid_volume.shape[1]*grid_volume.shape[2]))
            curr_min = min_grid_vol[i, j]
            curr_max = max_grid_vol[i, j]
            curr_std = std_grid_vol[i, j]
            curr_mean = mean_grid_vol[i, j]

            min_height = curr_mean + noise_check_ratio*np.abs(curr_mean - curr_min)
            peaks = scipy.signal.find_peaks(grid_volume[:, i, j], height=min_height, distance=10)

            if len(peaks[0]) == 0:
                continue
            else:
                #print('NUM PEAKS at {}, {} : {}'.format(i, j, len(peaks[0])))
                peaks_volume[peaks[0], i, j] = 1

                if continuous_threshold:
                    peaks_volume[:, i,j], filtered_inds = filter_continuous_peaks(peaks_volume[:, i, j], peaks[0], continuous_threshold)
                    #print('filtered:', len(filtered_inds))
    return peaks_volume

def filter_continuous_peaks(arr, peak_inds, continuous_threshold):
    curr_filter = 0
    filtered_inds = []
    for peak_i in sorted(peak_inds):
        if arr[peak_i] == 0:
            filtered_inds.append(peak_i)
            continue
        i = peak_i + 1
        curr_continuous = 0
        while (curr_continuous != continuous_threshold) and (i != len(arr)):
            if arr[i] == 1:
                curr_continuous = 0
                arr[i] = 0
            else:
                curr_continuous += 1
            i += 1
    return arr, filtered_inds

def is_signal_volume(grid_volume, noise_check_ratio):
    '''
    Takes in grid volume and determines time of signal per timeframe
    Params:
        grid_volume: the grid volume
        box_size:       the box_size
        noise_check_ratio: ...
    '''
    grid_volume = np.copy(grid_volume)
    #print('Obtaining statistics...')
    mean_grid, std_grid, min_grid, max_grid = get_grid_statistics(grid_volume)
    #print('Done')
    is_signal_vol = (np.abs(grid_volume - mean_grid[None, :]) /
                    np.abs(min_grid[None, :] - mean_grid[None, :])) >= noise_check_ratio
    return is_signal_vol

def get_activity_ratio_volume(grid_volume):
    activities = []
    for stack_i in range(grid_volume.shape[0]):
        activities.append(get_activity_ratio_grid(grid_volume[stack_i, :, :]))
    return np.array(activities)

def get_activity_ratio_grid(box_grid, activity_value=0):
    return len(np.where(box_grid > activity_value)[0]) / (box_grid.shape[0]*box_grid.shape[1])

def get_heatmap_volume(signal_volume):
    '''
    Takes in volume consisting of signals and generates volume that
    continiously appends the signals of previous frames
    '''
    heatmap_volume = np.zeros(signal_volume.shape)
    heatmap_volume[0, :, :] = signal_volume[0, :, :]
    for i in range(1, signal_volume.shape[0]):
        heatmap_volume[i, :, :] = heatmap_volume[i - 1, :, :] + signal_volume[i, :, :]
    return heatmap_volume

def apply_gauss_filter_grid_volume(grid_volume, sigma=10):
    grid_volume_gauss = np.zeros(grid_volume.shape)
    #Apply gaussian filter to signal
    for i in range(grid_volume.shape[1]):
        for j in range(grid_volume.shape[2]):
            grid_volume_gauss[:, i, j] = gaussian_filter(grid_volume[:, i, j], sigma=sigma)
    return grid_volume_gauss

def get_signal_length_from_threshold(signal, peak_index, threshold):
    '''
    Given grid volume and peak index, threshold returns length of signal
    '''
    s_low = peak_index
    s_high = peak_index

    s_min_reached = False
    s_max_reached = False
    #Move to left until threshold is reached
    while (not s_min_reached) and s_low != 0:
        if signal[s_low - 1] > threshold:
            s_low -= 1
        else:
            s_min_reached = True
    while (not s_max_reached) and s_high != len(signal)-1:
        if signal[s_high + 1] > threshold:
            s_high += 1
        else:
            s_max_reached = True

    return s_high - s_low + 1, s_low, s_high
