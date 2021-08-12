import numpy as np
import pandas
from copy import deepcopy
from preprocessing.utils import hampel
from scipy.ndimage.measurements import label
from scipy.ndimage.filters import gaussian_filter

def df_average(d, channels, n=2, to_sum_keys=['speed']):
    '''
    D = dictionary of dictionaries each corresponding to a different channel
    n = average over n values
    to_sum_keys = keys to take sum instead of mean. e.g. speed should take sum as it is total speed over new duration
    '''
    d = deepcopy(d)
    for channel in channels:
        for k in d[channel].keys():
            remove_last_n = -(len(d[channel][k]) % n)
            if remove_last_n != 0:
                d[channel][k] = d[channel][k][:-remove_last_n]
            if k in to_sum_keys:
                d[channel][k] = np.sum(d[channel][k].reshape(-1, n), axis=1)
            else:
                d[channel][k] = np.mean(d[channel][k].reshape(-1, n), axis=1)
    return d

# Stick preprocessing:
# Find indices with extreme values (based on hampel and above some mean and std) and set them to 1, rest leave to empty
def stick_preprocessing(arr, num_std=4):
    extra_threshold = np.mean(arr) + num_std * np.std(arr)
    stick_indices = np.where(arr > extra_threshold)[0]
    stick_bin = np.zeros([len(arr)])
    stick_bin[stick_indices] = 1
    return stick_bin

#Pupil preprocessing
# Set any indices with abnormal values to one specified by moving average over window (apply Hampel filter)
def pupil_preprocessing(arr, hampel_std=1, hampel_k=100):
    pupil_df = pandas.DataFrame(arr)
    pupil_values = np.array(hampel(pupil_df, k=hampel_k, num_std=hampel_std, only_indices=False)).flatten()
    return pupil_values

#Speed preprocessing
def speed_preprocessing(arr):
    speed_bin_arr = np.copy(arr)
    speed_bin_arr[speed_bin_arr > 0] = 1
    return speed_bin_arr

def whiskers_preprocessing(arr):
    #Convert to distance from mean
    whiskers_d = np.abs(arr - np.mean(arr))
    whiskers_proc = gaussian_filter(whiskers_d, sigma=10)
    whiskers_diff = np.abs(whiskers_proc[1:] - whiskers_proc[:-1])
    whisker_threshold = np.median(whiskers_diff) + 0.5*np.std(whiskers_diff)

    whiskers_final = np.zeros((whiskers_diff.shape[0]))
    whiskers_final[whiskers_diff > whisker_threshold] = 1
    whiskers_final = np.append(whiskers_final, [whiskers_final[-1]])

    return whiskers_final

def get_behaviour_indices(arr, value, ind_past=0, ind_future=0, complementary=True):
    indices = []
    for ind in np.where(arr == value)[0]:
        indices.extend(range(max(0, ind - ind_past), min(len(arr), ind+ind_future+1)))
    output = np.sort(np.array(list(set(indices))))

    if complementary:
        compl = np.setdiff1d(np.arange(0, len(arr)), output)
        return output, compl
    else:
        return output

def remove_ends(arr, num=1):
    '''
    Given np.array with values. remove ends from consecutives
    [1, 3, 4, 5, 6, 8] -> [1, 3, 4, 5, 8] for num = 1
    [1, 3, 4, 5, 6, 8] -> [1, 3, 4, 8] for num = 2
    '''
    consecutives_l = group_consecutives(arr)
    new_arr = []

    for consecutive in consecutives_l:
        if min(num, len(consecutive)-1) == 0:
            new_arr.append(consecutive[0])
        else:
            new_arr.extend(consecutive[:-min(num, len(consecutive)-1)])

    return np.sort(np.array(new_arr))

def remove_consecutives(arr, max_size=1, inverse=False):
    '''
    Given np array with values, remove increments
    [1, 3, 4, 5, 7, 8] -> [1, 3, 7]
    '''
    consecutives_l = group_consecutives(arr)
    consecutives_max_l = []

    for consecutive in consecutives_l:
        if len(consecutive) <= max_size:
            consecutives_max_l.extend(consecutive)
        else:
            if inverse:
                consecutives_max_l.extend(consecutive[-max_size:])
            else:
                consecutives_max_l.extend(consecutive[:max_size])
    return np.array(consecutives_max_l)

def group_consecutives(vals, step=1):
    """Return list of consecutive lists of numbers from vals (number list)."""
    run = []
    result = [run]
    expect = None
    for v in vals:
        if (v == expect) or (expect is None):
            run.append(v)
        else:
            run = [v]
            result.append(run)
        expect = v + step
    return result

def get_bin_transition_indices(arr):
    """
    Given binary array, return indices where array goes from 0->1 (at 1 mark) and from 1->0 (at 0 mark)
    """
    arr = np.copy(arr).astype(np.int32)
    grads = arr[1:] - arr[:-1]
    up = np.where(grads == 1)[0] + 1
    down = np.where(grads == -1)[0] + 1
    return up, down

def get_delay_to_threshold(arr, threshold):
    '''
    Return index at which threshold value is first breached given array
    '''
    if len(arr) == 0:
        return None

    threshold_ind = np.argmax(arr > threshold)

    if (threshold_ind == 0) and (arr[threshold_ind] < threshold):
        return None
    else:
        return threshold_ind

def get_delay_to_index(val, sorted_arr, max_delay=None):
    '''
    Get difference of closest (greater) value in sorted_arr and val.
    If difference > max_delay return None.

    '''
    if len(sorted_arr) == 0:
        return None

    if max_delay == None:
        max_delay = sorted_arr[-1]

    closest_index = np.searchsorted(sorted_arr, val)

    if (closest_index == len(sorted_arr)) and (sorted_arr[-1] < val):
        return None

    index_loc = sorted_arr[closest_index]
    distance = index_loc - val

    if distance > max_delay:
        return None
    else:
        return distance

def read_behaviour_indices_no_whiskers_axon_astro(behaviours_path, oscilloscope_path,
                            stick_threshold_std=2):
    """
    Given a behaviours_path .csv and oscilloscope_path .csv return
    relevant behavioural indices

    default                 - all frame indices

    running                 - (:20 frames) when mouse is running (not just start running)
    running_start           - (0:10 frames when mouse start running)
    running_expect          - (-10:0 frames) when mouse start running from rest
    running_exact           - (frames) when mouse is running
    no_running_exact        - (frames) when mouse is not running

    stick_exact_start       - (frames) when stick just hits the mouse
    stick_start             - (0:10 frames) around !first touch! on the stick
    stick_end               - (0:10 frames at end of stick)
    stick_expect            - (-10:0 frames) around !first touch! on the stick (see if mouse expects)

    rest                    - opposite of running
    stick_rest              - (frames) when stick is on mouse and mouse is resting
    """

    df_behaviours = pandas.read_csv(behaviours_path)
    df_oscilloscope = pandas.read_csv(oscilloscope_path)
    df_extra = pandas.concat([df_behaviours, df_oscilloscope], axis=1, sort=False)
    roi_dict = {'extra' : {k : df_extra[k].values for k in df_extra.columns}}

    #avg_num = 3

    #roi_dict = df_average(roi_dict, colour_channels, n=avg_num, to_sum_keys=['speed'])
    stick_bin = stick_preprocessing(roi_dict['extra']['stick'], num_std=stick_threshold_std)
    speed_bin = speed_preprocessing(roi_dict['extra']['speed'])
    speed_values = roi_dict['extra']['speed']
    pupil_values = pupil_preprocessing(roi_dict['extra']['pupil'])

    #running                 - (frames) when mouse is running
    #no running              - (frames) when mouse is not running)
    running_exact_ind, rest_exact_ind = get_behaviour_indices(speed_bin, 1, complementary=True)
    default_ind = np.arange(len(running_exact_ind) + len(rest_exact_ind))
    running_ind, rest_ind = get_behaviour_indices(speed_bin, 1, ind_past=0, ind_future=15, complementary=True)
    #running_expect          - (-10:0 frames) when mouse start running from rest
    running_expect_ind = np.sort(np.array(list(set(remove_consecutives(get_behaviour_indices(speed_bin, 1, ind_past=15, complementary=False), max_size=15)) - set(running_ind))))
    #running_exact_start     - (frames when mouse start running)
    running_exact_start_ind = remove_consecutives(get_behaviour_indices(speed_bin, 1, ind_past=0, ind_future=15, complementary=False))
    running_semi_exact_ind = remove_ends(get_behaviour_indices(speed_bin, 1, ind_past=0, ind_future=15, complementary=False), num=15)
    rest_semi_exact_ind = np.sort(np.array(list(set(default_ind) - set(running_semi_exact_ind))))

    stick_ind, no_stick_ind = get_behaviour_indices(stick_bin, 1, ind_past=0, ind_future=15, complementary=True)
    #stick_start_exact       - (frames) when stick just hits the mouse
    stick_exact_start_ind = remove_consecutives(get_behaviour_indices(stick_bin, 1, ind_future=15, complementary=False))
    #stick_start             - (-2:10 frames) around !first touch! on the stick
    stick_start_ind = remove_consecutives(get_behaviour_indices(stick_bin, 1, ind_past=0, ind_future=15, complementary=False), max_size=15)
    #stick_end               - (-3:10 frames at end of stick)
    stick_end_ind = remove_consecutives(get_behaviour_indices(stick_bin, 1, ind_past=0, ind_future=15, complementary=False), max_size=15, inverse=True)
    #stick_expect            - (-10:0 frames) around !first touch! on the stick (see if mouse expects)
    stick_expect_ind = remove_consecutives(get_behaviour_indices(stick_bin, 1, ind_past=15, ind_future=0, complementary=False), max_size=15)
    #stick_rest              - (frames) when stick is on mouse and mouse is resting
    stick_rest_ind = np.array(list(set(get_behaviour_indices(stick_bin, 1, complementary=False)) & set(rest_ind)))

    stick_run_ind_15 = np.array(list(set(get_behaviour_indices(stick_bin, 1, ind_future=15, complementary=False))  & set(running_ind)))
    stick_run_ind_30 = np.array(list(set(get_behaviour_indices(stick_bin, 1, ind_future=30, complementary=False)) & set(running_ind)))

    #running_start =         - (0, 15 frames after running starts), excluding stick start
    running_start_ind = np.sort(np.array(list(set(remove_consecutives(get_behaviour_indices(speed_bin, 1, ind_future=15, complementary=False), max_size=15)) - set(stick_start_ind))))
    rest_start_ind = np.sort(np.array(list(set(remove_consecutives(rest_ind))))) - 15 #-15 here is because running ind looks 15 frames in future
    #Just for the start
    rest_start_ind[rest_start_ind < 0] += 15
    #stick_rest_no_run_ind   - (frames) when mouse is on stick is resting and there is no running activity (running_ind) around.
    #                        - since running_ind include 15 frames in future, if its 0 1 0 0 1 0 0 0 0 0 0 0 0 ... 0 0,
    #                        - the frame 0, 2, 3 won't be included if 1 is running and 0 resting
    stick_rest_no_run_ind = np.array(list(set(np.copy(stick_rest_ind)) - set(running_ind)))

    nrr_ind, rnr_ind = get_bin_transition_indices(speed_bin)
    random_ind = np.random.randint(len(stick_bin), size=len(running_ind))
    random_ind2 = np.random.randint(len(stick_bin), size=len(running_ind))
    random_ind3 = np.random.randint(len(stick_bin), size=len(running_ind))

    print('Running ind', len(running_ind), running_ind[0:10])
    print('Rest ind', len(rest_ind), rest_ind[0:10])
    print('Running exact ind', len(running_exact_ind), running_exact_ind[0:10])
    print('Rest exact ind', len(rest_exact_ind), rest_exact_ind[0:10])
    print('No stick ind', len(no_stick_ind), no_stick_ind[0:10])

    print('Stick ind', len(stick_ind), stick_ind[0:10])
    print('Stick exact start ind ', len(stick_exact_start_ind), stick_exact_start_ind[0:10])

    print('Stick start ind', len(stick_start_ind), stick_start_ind[0:10])
    print('Stick end ind', len(stick_end_ind), stick_end_ind[0:10])
    print('Stick expect ind', len(stick_expect_ind), stick_expect_ind[0:10])
    print('Stick rest ind', len(stick_rest_ind), stick_rest_ind[0:10])

    print('Running exact start ind', len(running_exact_start_ind), running_exact_start_ind[0:10])
    print('Running start ind', len(running_exact_start_ind), running_exact_start_ind[0:10])

    print('nrr ind', len(nrr_ind), nrr_ind[0:10])
    print('rnr ind', len(rnr_ind), rnr_ind[0:10])

    indices_d = {'default' : default_ind,
                 'rest_exact' : rest_exact_ind,
                 'no_stick' : no_stick_ind,

                 'stick' : stick_ind,
                 'stick_exact_start' : stick_exact_start_ind,
                 'stick_start' : stick_start_ind,
                 'stick_end'   : stick_end_ind,
                 'stick_expect' : stick_expect_ind,

                 'running' : running_ind,
                 'running_exact' : running_exact_ind,
                 'running_semi_exact' : running_semi_exact_ind,
                 'running_start' : running_start_ind,
                 'running_exact_start' : running_exact_start_ind,
                 'running_before' : running_expect_ind,

                 'stick_run_ind_15' : stick_run_ind_15,
                 'stick_run_ind_30' : stick_run_ind_30,

                  'stick_rest_no_run' : stick_rest_no_run_ind,

                 'rest' : rest_ind,
                 'rest_semi_exact' : rest_semi_exact_ind,
                 'rest_start' : rest_start_ind,
                 'stick_rest' : stick_rest_ind,

                 'rnr' : rnr_ind,
                 'nrr_ind' : nrr_ind,

                 'random_1' : random_ind,
                 'random_2' : random_ind2,
                 'random_3' : random_ind3,
                }

    remove_keys = []
    for k in indices_d.keys():
        #print('ELEN?', len(indices_d[k]))
        if len(indices_d[k]) == 0:
            remove_keys.append(k)

    for k in remove_keys:
        print('REMOVED {} key'.format(k))
        del indices_d[k]

    return indices_d, roi_dict, [stick_bin, speed_bin, pupil_values, speed_values]
    #return indices_d, roi_dict, [stick_bin, speed_bin, pupil_values]

def bin_avg_np_array(np_arr, bin_size=1):
    if bin_size == 1 or bin_size == None:
        return np_arr

    num_bins = int(len(np_arr) // bin_size)
    bin_mod = len(np_arr) % bin_size 

    if bin_mod != 0:
        np_arr = np_arr[:-bin_mod]
    
    return np.array([np.mean(binned_arr) for binned_arr in np.split(np_arr, num_bins)])

def read_behaviour_indices(behaviours_path, oscilloscope_path,
                            stick_threshold_std=2,
                            avg_extra=1, continuous_stick=False):
    """
    Given a behaviours_path .csv and oscilloscope_path .csv return
    relevant behavioural indices

    default                 - all frame indices

    running                 - (:15 frames) when mouse is running (not just start running)
    running_start           - (0:15 frames when mouse start running)
    running_expect          - (-15:0 frames) when mouse start running from rest
    running_exact           - (frames) when mouse is running
    no_running_exact        - (frames) when mouse is not running

    stick_exact_start       - (frames) when stick just hits the mouse
    stick_start             - (0:15 frames) around !first touch! on the stick
    stick_end               - (0:15 frames at end of stick)
    stick_expect            - (-15:0 frames) around !first touch! on the stick (see if mouse expects)

    rest                    - opposite of running
    stick_rest              - (frames) when stick is on mouse and mouse is resting
    """

    df_behaviours = pandas.read_csv(behaviours_path)
    df_oscilloscope = pandas.read_csv(oscilloscope_path)
    
    df_extra = pandas.concat([df_behaviours, df_oscilloscope], axis=1, sort=False)
    if (avg_extra == 1) or (avg_extra is None):
        roi_dict = {'extra' : {k : df_extra[k].values for k in df_extra.columns}}
    else:
        roi_dict = {'extra' : {k : bin_avg_np_array(df_extra[k].values, bin_size=avg_extra) for k in df_extra.columns}}

    #avg_num = 3
    #roi_dict = df_average(roi_dict, colour_channels, n=avg_num, to_sum_keys=['speed'])
    stick_values = roi_dict['extra']['stick']
    stick_bin = stick_preprocessing(roi_dict['extra']['stick'], num_std=stick_threshold_std)

    # Will look at up to 1.5 seconds (if 10.3 frames per second as default) after event
    # e.g. if there is running at index 1, will look at indices 1-16 as running
    # running_exact can be used for exact indices instead

    ind_future_num = 15

    if avg_extra > 1:
        ind_future_num = int(ind_future_num / avg_extra)

    print('IND FUTURE NUM', ind_future_num)
    # If continuous stick we set everything to 1
    if continuous_stick:
        stick_bin = [1 for v in stick_bin]
    speed_bin = speed_preprocessing(roi_dict['extra']['speed'])
    speed_values = roi_dict['extra']['speed']
    whisker_bin = whiskers_preprocessing(roi_dict['extra']['whiskers'])

    pupil_values = pupil_preprocessing(roi_dict['extra']['pupil'])

    #running                 - (frames) when mouse is running
    #no running              - (frames) when mouse is not running)
    running_exact_ind, rest_exact_ind = get_behaviour_indices(speed_bin, 1, complementary=True)
    default_ind = np.arange(len(running_exact_ind) + len(rest_exact_ind))
    running_ind, rest_ind = get_behaviour_indices(speed_bin, 1, ind_past=0, ind_future=ind_future_num, complementary=True)
    #when mouse is running but removing the last 15 frames of each run (when it stops exactly)
    running_semi_exact_ind = remove_ends(get_behaviour_indices(speed_bin, 1, ind_past=0, ind_future=ind_future_num, complementary=False), num=ind_future_num)
    rest_semi_exact_ind = np.sort(np.array(list(set(default_ind) - set(running_semi_exact_ind))))
    #running_expect          - (-10:0 frames) when mouse start running from rest
    running_expect_ind = np.sort(np.array(list(set(remove_consecutives(get_behaviour_indices(speed_bin, 1, ind_past=ind_future_num, complementary=False), max_size=ind_future_num)) - set(running_ind))))
    #running_exact_start           - (exact frames when mouses just starts running)
    running_exact_start_ind = remove_consecutives(get_behaviour_indices(speed_bin, 1, ind_past=0, ind_future=ind_future_num, complementary=False))
    stick_ind, no_stick_ind = get_behaviour_indices(stick_bin, 1, ind_past=0, ind_future=ind_future_num, complementary=True)



    #stick_start_exact       - (frames) when stick just hits the mouse
    stick_exact_start_ind = remove_consecutives(get_behaviour_indices(stick_bin, 1, ind_future=ind_future_num, complementary=False))
    #stick_start             - (0:10 frames) around !first touch! on the stick
    stick_start_ind = remove_consecutives(get_behaviour_indices(stick_bin, 1, ind_past=0, ind_future=ind_future_num, complementary=False), max_size=ind_future_num)
    #stick_end               - (0:10 frames at end of stick)
    stick_end_ind = remove_consecutives(get_behaviour_indices(stick_bin, 1, ind_past=0, ind_future=ind_future_num, complementary=False), max_size=ind_future_num, inverse=True)
    #stick_expect            - (-10:0 frames) around !first touch! on the stick (see if mouse expects)
    stick_expect_ind = remove_consecutives(get_behaviour_indices(stick_bin, 1, ind_past=ind_future_num, ind_future=0, complementary=False), max_size=ind_future_num)
    #stick_rest              - (frames) when stick is on mouse and mouse is resting
    stick_rest_ind = np.array(list(set(get_behaviour_indices(stick_bin, 1, complementary=False)) & set(rest_ind)))
    #stick run               - (frames) when stick hits the mouse and x frames after when its only running
    stick_run_ind_15 = np.array(list(set(get_behaviour_indices(stick_bin, 1, ind_future=ind_future_num, complementary=False))  & set(running_ind)))
    stick_run_ind_30 = np.array(list(set(get_behaviour_indices(stick_bin, 1, ind_future=ind_future_num*2, complementary=False)) & set(running_ind)))

    #running_no_stick - when its running but not after having hit stick (up to 15 frames)
    running_no_stick_ind = np.array(list(set(running_ind) - set(stick_run_ind_15)))

    #stick_rest_no_run_ind   - (frames) when mouse is on stick is resting and there is no running activity (running_ind) around.
    #                        - since running_ind include 15 frames in future, if its 0 1 0 0 1 0 0 0 0 0 0 0 0 ... 0 0,
    #                        - the frame 0, 2, 3 won't be included if 1 is running and 0 resting
    stick_rest_no_run_ind = np.array(list(set(np.copy(stick_rest_ind)) - set(running_ind)))

    whisker_exact_ind, no_whisker_exact_ind = get_behaviour_indices(whisker_bin, 1, ind_past=0, ind_future=0, complementary=True)
    whisker_ind, no_whisker_ind = get_behaviour_indices(whisker_bin, 1, ind_past=0, ind_future=ind_future_num, complementary=True)

    #running_start =         - (0, 15 frames after running starts), excluding stick start
    running_start_ind = np.sort(np.array(list(set(remove_consecutives(get_behaviour_indices(speed_bin, 1, ind_future=ind_future_num, complementary=False), max_size=ind_future_num)) - set(stick_start_ind))))

    rest_start_ind = np.sort(np.array(list(set(remove_consecutives(rest_ind))))) - ind_future_num #-15 here is because running ind looks 15 frames in future
    #Just for the start
    rest_start_ind[rest_start_ind < 0] += ind_future_num

    #Whiskering at rest at stick
    whisker_rest_stick_ind = np.array(list(set(whisker_ind) & set(stick_ind) & set(rest_ind)))
    #Whiskering at stick
    whisker_stick_ind = np.array(list(set(whisker_ind) & set(stick_ind)))

    nrr_ind, rnr_ind = get_bin_transition_indices(speed_bin)
    random_ind = np.random.randint(len(stick_bin), size=len(running_ind))
    random_ind2 = np.random.randint(len(stick_bin), size=len(running_ind))
    random_ind3 = np.random.randint(len(stick_bin), size=len(running_ind))

    print('Running ind', len(running_ind), running_ind[0:10])
    print('Rest ind', len(rest_ind), rest_ind[0:10])
    print('Running exact ind', len(running_exact_ind), running_exact_ind[0:10])
    print('Rest exact ind', len(rest_exact_ind), rest_exact_ind[0:10])
    print('No stick ind', len(no_stick_ind), no_stick_ind[0:10])

    print('Stick ind', len(stick_ind), stick_ind[0:10])
    print('Stick exact start ind ', len(stick_exact_start_ind), stick_exact_start_ind[0:10])
    print('Stick start ind', len(stick_start_ind), stick_start_ind[0:10])
    print('Stick end ind', len(stick_end_ind), stick_end_ind[0:10])
    print('Stick expect ind', len(stick_expect_ind), stick_expect_ind[0:10])
    print('Stick rest ind', len(stick_rest_ind), stick_rest_ind[0:10])

    print('Running exact start ind', len(running_exact_start_ind), running_exact_start_ind[0:10])

    print('nrr ind', len(nrr_ind), nrr_ind[0:10])
    print('rnr ind', len(rnr_ind), rnr_ind[0:10])

    indices_d = {'default' : default_ind,
                 'rest_exact' : rest_exact_ind,
                 'no_stick' : no_stick_ind,

                 'stick' : stick_ind,
                 'stick_exact_start' : stick_exact_start_ind,
                 'stick_start' : stick_start_ind,
                 'stick_end'   : stick_end_ind,
                 'stick_expect' : stick_expect_ind,

                 'running' : running_ind,
                 'running_exact' : running_exact_ind,
                 'running_semi_exact' : running_semi_exact_ind,
                 'running_exact_start' : running_exact_start_ind,
                 'running_start' : running_start_ind,
                 'running_before' : running_expect_ind,
                 'running_no_stick' : running_no_stick_ind,
                 'stick_run_ind_15' : stick_run_ind_15,
                 'stick_run_ind_30' : stick_run_ind_30,

                 'stick_rest_no_run' : stick_rest_no_run_ind,

                 'rest' : rest_ind,
                 'rest_semi_exact' : rest_semi_exact_ind,
                 'rest_start' : rest_start_ind,
                 'stick_rest' : stick_rest_ind,
                 #'running_start_whisker' : running_start_whisker_ind,
                 #'running_start_no_whisker' : running_start_no_whisker_ind,

                'whisker_exact' : whisker_exact_ind,
                'no_whisker_exact' : no_whisker_exact_ind,
                'whisker' : whisker_ind,
                'no_whisker' : no_whisker_ind,
                'whisker_rest_stick' : whisker_rest_stick_ind,
                'whisker_stick' : whisker_stick_ind,

                 'rnr' : rnr_ind,
                 'nrr_ind' : nrr_ind,

                 'random_1' : random_ind,
                 'random_2' : random_ind2,
                 'random_3' : random_ind3,
                }

    remove_keys = []
    for k in indices_d.keys():
        #print('ELEN?', len(indices_d[k]))
        if len(indices_d[k]) == 0:
            remove_keys.append(k)

    for k in remove_keys:
        print('REMOVED {} key'.format(k))
        del indices_d[k]

    return indices_d, roi_dict, [stick_bin, speed_bin, whisker_bin, stick_values, speed_values, pupil_values]

def get_connected_components_2D(array, structure=np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])):
    '''
    Given 2D array of values, find connected components of non-zero values
    in array
    '''
    labeled, ncomponents = label(array, structure)
    return labeled, ncomponents

def get_component_info(grid):
    labeled, ncomponents = get_connected_components_2D((grid > 0).astype(int))
    sizes_l = []
    for component_i in range(1, ncomponents+1):
        sizes_l.append(np.sum(labeled == component_i))
    return labeled, ncomponents, sizes_l
