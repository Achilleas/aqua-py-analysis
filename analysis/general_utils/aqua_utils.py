import numpy as np
import h5py
import os, sys, glob
import scipy

def read_aqua_mats(mats_folder_path):
    '''
    Args:
        Folder path consisting of .mat files extracted from AQUA. This process
        concatenates all the features in the .mat files.
    Returns
        res_d (dict): dictionary of concatenated features from mat files.
            Contains the following keys:
            ['peri', 'area', 'circMetric', 'decayTau', 'dffMax', 'dffMax2',
            'dffMaxFrame', 'dffMaxPval', 'dffMaxZ', 'fall91', 'rgt1', 'rise19',
            'tBegin', 'tEnd', 'width11', 'width55', 'areaChange',
            'areaChangeRate', 'areaFrame', 'propGrow', 'propGrowOverall',
            'propShrink', 'propShrinkOverall', 't0', 't1', 'x2D', 'x3D',
            'border_mask', 'clandmark_mask', 'clandmark_center',
            'border_dist2border', 'clandmark_distAvg', 'dff_raw', 'dff_only',
            'd_raw', 'd_only', 'event_i_video_segment', 'time', 'time_s', 'duration']
            Take a look in AQUA outputs parameters:
            https://drive.google.com/file/d/1U3oJpEFwv0lXdax6efSnoifcYjJuRzj3/view
            for more details

        input_shape_l (list): list of input shapes in form (n_frames, d_1, d_2)
        framerates_inv_l : framerate
        spatial_res: spatial resolution
    Extract from each mat file in folder:

    res_d, input_shape_l, framerates_inv_l[0], spatial_res_l[0]

    Mat file structure:
    ----------------------------------------------------------------------------
    % Basic - Area                          | res.fts.basic.area
    % Basic - Perimeter                     | res.fts.basic.peri
    % Basic - Circularity                   | res.fts.basic.circMetric
    % Curve - P Value on max Dff (-log10)   | -log10(res.fts.curve.dffMaxPval)
    % Curve - Max Dff                       | res.fts.curve.dffMax
    % Curve - Duration 50% to 50%           | res.fts.curve.width55
    % Curve - Duration 10% to 10%           | res.fts.curve.width11
    % Curve - Rising duration 10% to 90%    | res.fts.curve.rise19
    % Curve - Decaying duration 90% to 10%  | res.fts.curve.fall91
    % Curve - Decay tau                     | res.fts.curve.decayTau
    % Curve - tBegin (frame num)            | res.fts.curve.tBegin (349 index 1)
    % Curve - tEnd   (frame num)            | res.fts.curve.tEnd (363 index 1)
    % Curve - rgt1 (extended start end)     | res.fts.curve.rgt1 (348, 364 index 1)

    % Propagation - onset - overall         | sum(res.fts.propagation.propGrowOverall)
    % Propagation - onset - one dir Anterior| sum(res.fts.propagation.propGrow{i}(:, 1)

    % Spatial location of event             | res.fts.loc.x2D{i}
    % Spatiotemporal location of event      | res.fts.loc.x3D{i}
    % start time at 20% max intensity       | res.fts.loc.t0(i) (349 index 1)
    % end time at 20% max intensity         | res.fts.loc.t1(i) (363 index 1)

    % LANDMARKS
    % res.fts.region.landMark
         mask: {[412×412 double]}
         center: [238.4537 210.1019]
         border: {[38×2 double]}
        centerBorderAvgDist: 7.5235

    % res.fts.region.landmarkDist.distAvg
        dist avg

    % mask of landmark res.fts.region.landMark.mask
    % center of mask of landmark center: [238.4537 210.1019] res.fts.region.landMark.center: [238.4537 210.1019]
    % cell mask res.fts.region.cell.mask
    % cell distance to border for each event dist2border
    ----------------------------------------------------------------------------
    '''
    res_d = {}

    framerates_inv_l = []
    spatial_res_l = []
    input_shape_l = []
    if len(glob.glob(os.path.join(mats_folder_path, '*.mat'))) == 0:
        print('.mat files dont exist here: {}'.format(mats_folder_path))
        return None

    for filepath in sorted(glob.glob(os.path.join(mats_folder_path, '*.mat'))):
        print(filepath)

        with h5py.File(filepath, 'r') as matfile:
            d = {}
            input_shape = np.copy(np.array(matfile['res']['datOrg'])).transpose().shape
            input_shape = np.array(input_shape, dtype=np.uint64)
            input_shape_l.append(input_shape)

            opts_d = matfile['res']['opts']
            basic_d = matfile['res']['fts']['basic']
            curve_d = matfile['res']['fts']['curve']
            propagation_d = matfile['res']['fts']['propagation']
            loc_d = matfile['res']['fts']['loc']

            if len(matfile['res']['fts']['region']) == 4:
                region_d = matfile['res']['fts']['region']
            elif len(matfile['res']['fts']['region']) == 2:
                print('NO REGION')
            else:
                print('Region len: ', len(matfile['res']['fts']['region']))
            #print('Opts', opts_d)
            framerates_inv_l.append(float(opts_d['frameRate'][0][0]))
            spatial_res_l.append(float(opts_d['spatialRes'][0][0]))

            indices_keys = ['tBegin', 'tEnd', 'rgt1', 't0', 't1', 'dffMaxFrame']
            no_append_keys = ['border_mask', 'clandmark_mask', 'clandmark_center']
            index_append = 0
            index_append_3D = 0

            index_append = np.sum([i_shape[-1] for i_shape in input_shape_l]) - input_shape_l[0][-1]
            index_append_3D = np.sum([np.prod(i_shape) for i_shape in input_shape_l]) - np.prod(input_shape_l[0])

            #print('index append:', index_append)
            #print('3D index append', index_append_3D)
            #Basic (exclude map)
            for k in list(set(basic_d.keys()) - set(['map'])):
                d[k] = basic_d[k]
            #Curve
            for k in curve_d.keys():
                d[k] = curve_d[k]

            #Propagation
            for k in propagation_d.keys():
                d[k] = propagation_d[k]
                if k == 'areaChange' or \
                   k == 'areaChangeRate' or \
                   k == 'areaFrame' or \
                   k == 'propGrow' or \
                   k == 'propShrink':
                   l = []
                   for i in range(len(propagation_d[k])):
                        prop_d_np = np.array(matfile[propagation_d[k][i][0]]).squeeze().transpose()
                        l.append(prop_d_np)
                d[k] = np.array(l)

            #Locations
            for k in loc_d.keys():
                if k == 'x3D' or k == 'x2D':
                    l = []
                    for i in range(len(loc_d[k])):
                        loc_d_np = np.array(matfile[loc_d[k][i][0]], dtype=np.int64).squeeze().transpose()
                        #index-1 (matlab to python)
                        if k == 'x2D':
                            l.append(loc_d_np - 1)
                            #inds_2D = np.unravel_index(loc_d_np - 1, shape_2D, order='F')
                            #l.append(inds_2D)
                        #index-1 (matlab to python) and index append 3D to for stacking
                        elif k == 'x3D':
                            l.append(loc_d_np - 1 + index_append_3D)
                    d[k] = np.array(l)
                else:
                    d[k] = loc_d[k]

            #Regions & Landmarks if available in regions
            if len(matfile['res']['fts']['region']) == 4:
                #Just for first file:
                d['border_mask'] = np.array(matfile[region_d['cell']['mask'][0][0]])
                #Regions
                d['border_dist2border'] = region_d['cell']['dist2border']
                #Landmarks
                try:
                    d['clandmark_mask'] = np.array(matfile[region_d['landMark']['mask'][0][0]])
                    d['clandmark_center'] = region_d['landMark']['center']
                    d['clandmark_distAvg'] = region_d['landmarkDist']['distAvg']
                except Exception as err:
                    print('Met error with landmarks (skipping):', err)    
            #Curves and Df/f curves
            d['dff_raw'] = np.array(matfile['res']['dffMat'])[0]
            d['dff_only'] = np.array(matfile['res']['dffMat'])[1]

            d['d_raw'] = np.array(matfile['res']['dMat'])[0]
            d['d_only'] = np.array(matfile['res']['dMat'])[1]

            d['event_i_video_segment'] = np.zeros([d['d_raw'].shape[1]])
            d['event_i_video_segment'][:] = len(input_shape_l)

            for k in d.keys():
                d[k] = np.copy(np.array(d[k]).squeeze().transpose())

            if len(d['t0'].shape) == 0:
                continue

            for k in d.keys():

                #dffMax frame is incorrectly given in seconds not frames
                #convert seconds to frames indices
                if k == 'dffMaxFrame':
                    d[k] = get_aqua_frame_from_seconds_np(d[k], framerates_inv_l[-1])
                #index-1 (matlab to python), append index_append for stacking
                #Also convert to uint64 array as they are indices
                if k in indices_keys:
                    d[k] = d[k] - 1 + index_append
                    d[k] = d[k].astype(np.uint64)                

                #First instance
                if k not in res_d:
                    res_d[k] = d[k]
                #Not first instance but not concatenating
                elif (k in res_d) and (k in no_append_keys):
                    continue
                #Not first instance, concatenating on first axis
                else:
                    res_d[k] = np.concatenate([res_d[k], d[k]], axis=0)
    #print(len(framerates_inv_l))
    #print(len(spatial_res_l))
    return res_d, input_shape_l, framerates_inv_l[0], spatial_res_l[0]

def get_aqua_frame_from_seconds(s, fr_inv):
    '''
    Args:
        s : Number of seconds
        fr_inv : Seconds per framerate
    Returns:
        number of frames from seconds
    '''
    return int(np.round(s / fr_inv))

def get_aqua_frame_from_seconds_np(s_np, fr_inv, dtype=np.uint64):
    '''
    Args:
        s : Number of seconds
        fr_inv : Seconds per framerate
    Returns:
        number of frames from seconds
    '''
    frames_np = np.rint(s_np / fr_inv).astype(dtype)
    return frames_np

def get_event_grid_from_x2D(x2D, shape_2D):
    '''
    Args:
        x2D: list of lists with each sublist consisting of the flat array
            indices representing the spatial location of an event (at maximal cover)
        shape_2D:
            the shape of the area, to unravel the flat indices to 2D form
    Returns:
        Grid of occurences in area, corresponding to number of events that took place
    '''
    event_grid = np.zeros(shape_2D)

    for i in range(len(x2D)):
        event_locs = np.unravel_index((x2D[i]), shape_2D, order='F')
        event_grid[event_locs] += 1
    return event_grid

def get_dff_grid_from_x2D(x2D, dff, shape_2D):
    '''
    Same as get_event_grid_from_x2D but add their df/f values instead of count
    '''
    dff_grid = np.zeros(shape_2D)
    for i in range(len(x2D)):
        event_locs = np.unravel_index((x2D[i]), shape_2D, order='F')
        dff_grid[event_locs] += np.sum(dff[i])
    return dff_grid

def apply_res_d_filter(res_d, min_event_duration=2, min_um_size=2):
    '''
    Args:
        res_d (dict) : Features dictionary
        min_event_duration (int) : Minimum event duration (frames)
        min_um_size (int) : Minimum size in (um)
    Returns:
        Filtered res_d
    Apply filters on all events

    Note it doesn't copy, it changes the res_d dictionary
    '''
    #Filter 1
    duration_filtered = np.where((res_d['tEnd'] - res_d['tBegin']) >= min_event_duration)[0]

    #Filter 2
    size_filtered = np.where(res_d['area'] >= min_um_size)[0]

    filtered_inds = np.array(list(set(list(duration_filtered)) & set(list(size_filtered))))
    old_shape = len(res_d['area'])

    print('Filtered out events: FROM: {} TO: {}'.format(old_shape, len(filtered_inds)))

    for k in res_d.keys():
        if (res_d[k].shape[0] == old_shape):
            res_d[k] = res_d[k][np.array(filtered_inds)]
    return res_d


def get_delay_info_from_res(inds, res_d, event_inds_subset, min_delay=0,
                                max_delay=50, max_duration=None,
                                unique_events=True, return_event_ids=False,
                                return_non_unique_delays_arr=False,
                                min_duration=None):
    '''
    Args:
        inds : the frames we use as start points
        res_d : feature dictionary
        event_inds_subset (np.ndarray) : list of events to search for delays
        min_delay (int) : minimum event delay in frames from index
        max_delay (int) : maximum event delay in frames from index
        unique_events : if True return minimum delay for each index.
            e.g. index 512 has delays [2, 3,7, 24]
            for  min_delay=0, max_delay=30, only take [2]
        return_event_ids : return list of events
        return_non_unique_delays_arr :
        min_duration (int) : minimum duration of events to include

    Returns:
        ie_mins : delays to start of event signal
        peak_mins : delays to peak of event signal (not implemented for non-unique)
        valid_event_i : list of event ids corresponding to delays (must set return_event_ids=True)
        ie_mins_l_l : (return_non_unique_delays_arr) list of lists. event delay list for each index
        peak_mins_l_l : (return_non_unique_delays_arr) list of lists. event peak delay list for each index
        valid_event_i_l_l : (return_non_unique_delays_arr) list of lists. event id list for each index

        Given some indices, an event subset and a range of delay, obtain
        a list of events that take place within this delay from each index.
        Then return all the delays.
    '''
    signal_delays_l = []
    peak_delays_l = []

    t_begin_np = res_d['tBegin'][event_inds_subset]
    t_end_np = res_d['tEnd'][event_inds_subset]
    t_max_frame_np = res_d['dffMaxFrame'][event_inds_subset]
    t_duration_np = t_end_np - t_begin_np

    if max_duration is not None:
        filter_dur_i = np.where(t_duration_np <= max_duration)
        t_begin_np = t_begin_np[filter_dur_i]
        t_end_np = t_end_np[filter_dur_i]
        t_duration_np = t_duration_np[filter_dur_i]
        t_max_frame_np = t_max_frame_np[filter_dur_i]

    if min_duration is not None:
        filter_dur_i = np.where(t_duration_np >= min_duration)
        t_begin_np = t_begin_np[filter_dur_i]
        t_end_np = t_end_np[filter_dur_i]
        t_duration_np = t_duration_np[filter_dur_i]
        t_max_frame_np = t_max_frame_np[filter_dur_i]

    #Create 2D index-event grid [ind, num_events]
    ie_grid = np.zeros([len(inds), len(t_begin_np)])

    #Fill grid with t_begin values and remove index on every column
    ie_grid += t_begin_np[np.newaxis, :]
    ie_grid -= np.array(inds)[:, np.newaxis]

    #Set invalid values to max_delay + 1
    ie_grid[(ie_grid > max_delay) | (ie_grid < min_delay)] = max_delay + 1

    #One delay for each event. index 1 and 2 will have only delay 1 frame for event at frame 3
    if unique_events:
        #Find minimum over all indices (frames). As in what is the minimum delay for each event
        ie_argmins = np.argmin(ie_grid, axis=0)
        #Should be same as ie_grid[ie_argmins, :]
        ie_mins = ie_grid[ie_argmins, np.arange(len(ie_argmins))]
        #ie_mins = np.min(ie_grid, axis=0)
        #Filter out invalid values
        valid_event_i = np.where(ie_mins <= max_delay)
        ie_mins = ie_mins[valid_event_i]
        peak_mins = ie_mins + (t_max_frame_np[valid_event_i] - t_begin_np[valid_event_i])

    #Can have multiple delays for each event. e.g. index 1 and 2 will have delays
    #2 and 1 frames for event at frame 3
    else:
        valid_event_i = []
        ie_mins = []
        peak_mins = []

        ie_mins_l_l = []
        peak_mins_l_l = []
        valid_event_i_l_l = []

        for ind in range(ie_grid.shape[0]):
            valid_event_ind_i = np.where(ie_grid[ind, :] <= max_delay)
            ie_mins_v = ie_grid[ind, valid_event_ind_i]
            peak_mins_v = ie_mins_v +  (t_max_frame_np[valid_event_ind_i] - t_begin_np[valid_event_ind_i])

            ie_mins.extend(list(ie_mins_v.flatten().astype(int)))
            peak_mins.extend(list(peak_mins_v.flatten().astype(int)))
            valid_event_i.extend(list(valid_event_ind_i[0]))

            ie_mins_l_l.append(list(ie_mins_v.flatten().astype(int)))
            peak_mins_l_l.append(list(peak_mins_v.flatten().astype(int)))
            valid_event_i_l_l.append(list(valid_event_ind_i[0]))

        #ie_mins = ie_grid[np.where(ie_grid <= max_delay)].flatten()
        #print('NOT IMPLEMENTED FULLY')
        ie_mins = np.array(ie_mins)
        peak_mins = np.array(peak_mins)

    if return_event_ids:
        return ie_mins, peak_mins, valid_event_i

    if return_non_unique_delays_arr and unique_events == False:
        return ie_mins, peak_mins, valid_event_i, ie_mins_l_l, peak_mins_l_l, valid_event_i_l_l
    return ie_mins, peak_mins

def get_event_subsets(indices_d, res_d, after_i=0, before_i=0, to_print=False, return_info=False):
    '''
    Given dictionary indices, get all events taking place some range from each index
    If we want events after 20 frames of index set after_i=20.
    If we want events before 20 frames of index set before_i=20.
    Args:
        indices_d (dict) : dictionary of indices (keys tend to be the behaviours)
        res_d (dict) : Features dictionary
        after_i : max range after current index to look for events
        before_i : max range before current index to look for events
        to_print : stdout some details
        return_info : return extra details (look in code)
    Returns:
        event_subsets (dict): Dictionary of events filling criteria
                            (dictionary keys corresponding to indices_d keys)
    '''
    total_events = len(res_d['tBegin'])
    total_indices = len(indices_d['default'])
    behavioural_max_delay_to_event = 0

    indices_events_bin = np.zeros([total_indices, total_events], dtype=np.bool)
    event_subsets = {}

    #EVENT INDICES MAP
    event_indices_map_l = []
    for event_i in range(total_events):
        t_begin_i = res_d['tBegin'][event_i]
        event_indices = np.arange(max(0, t_begin_i - after_i), min(total_indices, t_begin_i+1+before_i)).astype(np.uint64)
        indices_events_bin[event_indices, event_i] = True

    for k in indices_d:
        #This gives us the subset of indices (ind_sub, total_events)
        indices_k_subset = indices_events_bin[indices_d[k], :]
        #Take sum over ind_sub
        indices_k_subset_sum = np.sum(indices_k_subset, axis=(0))
        #All non-0 values of array (total_events) are events that took place at some index of the subset of indices
        #Take the indices of the non-0 values of array
        event_subset = np.nonzero(indices_k_subset_sum)[0]
        event_subsets[k] = event_subset

        if to_print:
            print('Number of events: {} - {} ind_size {}'.format(k , len(event_subset), len(indices_d[k])))

    if return_info:
        return event_subsets, indices_events_bin
    return event_subsets

def get_event_centroids_from_x2D(x2D, shape_2D):
    '''
    Args:
        x2D (list) : Event locations in flattened indices
        shape_2D (np.ndarray) : the shape of the area
    Returns:
        list of event centroids for each event in x2D
    '''
    centroids_l = []
    for i in range(len(x2D)):
        event_inds = np.unravel_index((x2D[i]), shape_2D, order='F')
        centroids_l.append((int(np.floor(np.mean(event_inds[0]))), int(np.floor(np.mean(event_inds[1])))))
    return centroids_l

def get_euclidean_distances(xy, xy_l):
    '''
    Args:
        xy : tuple or array consisting of [x, y] value
        xy_l : list to xy
    Returns:
        dist: eucledian distances
    '''
    return scipy.spatial.distance.cdist(np.atleast_2d(xy), np.atleast_2d(xy_l))[0]

def radius_event_extraction(r_distances, center, border_mask, n_bins):
    '''
    Given array of r_distances, split into n bins of distances
    corresponding to the radius. We then take circles of each radius
    and calculate the area they consist removing circles of pixels of previous smaller
    r and pixels outside of border.
    Args:
        r_distances : the radial distances
        center : the center coordinates
        border_mask : the border mask
        n_bins : number of bins
    Returns:
        n_events_arr_norm: number of r_distances within first interval of r normalized by number of pixels
        n_events_i_arr: indices of r_distances on each interval (list of size n_bins with each element being a list of number of r_distances)
        area_bins: list of areas
        r_bins: list of radius bins
    '''
    #Bin distances (r) into n bins  & determine the number of events on each bin
    n_events_arr, r_bins = np.histogram(r_distances, bins=n_bins, range=(0, np.max(r_distances)))

    n_events_i_arr = []
    area_bins = []
    #Find indices of events of each bin
    sorted_i = np.argsort(r_distances)
    r_distances_s = r_distances[sorted_i]
    min_ind = 0
    for r_curr in r_bins[1:]:
        #Find where max value of r given bin_v is + 1 (this is our max range)
        max_ind = np.searchsorted(r_distances_s, r_curr, side='right')
        n_events_i_arr.append(np.array(sorted_i[min_ind:max_ind]))
        min_ind = max_ind

    #Find list of areas for each r bin (excluding previous r circle)
    r_prev = 0
    for r_curr in r_bins[1:]:
        area_bins.append(np.pi*(r_curr**2) - np.pi*(r_prev**2))
        r_prev = r_curr

    #For each area, remove ignored areas from border mask
    ignore_coordinates = np.transpose(np.where(border_mask==0))
    #IMPORTANT: GOTTA FLIP THE CENTER x,y to correspond with border mask coords (arr[y][x])
    center_flipped = [center[1], center[0]]
    ignore_r_arr = get_euclidean_distances(center_flipped, ignore_coordinates)
    ignore_r_arr_s = np.sort(ignore_r_arr)
    min_ind = 0
    for i, r_curr in enumerate(r_bins[1:]):
        max_ind = np.searchsorted(ignore_r_arr_s, r_curr, side='right')
        area_bins[i] -= (max_ind - min_ind)
        min_ind=max_ind
    #Normalize number of events by their corresponding area
    n_events_arr_norm = np.array(n_events_arr) / np.array(area_bins)

    return n_events_arr_norm, n_events_i_arr, area_bins, r_bins

def filter_range_inds(ind_ref, ind_source, range=(0, 15), prop=1.0):
    '''
    For example ind_ref = 'stick_exact_start' indices list
                ind_source = 'running' indices list
                range = 0,15
                prop = 0.8

    We find all values of ind_ref. For each value ind_ref we take
    range(ind_ref+range[0], ind_ref+range[1]) indices. Then compare if ind_source
    contains at least (prop) amount of these indices

    If prop = 1.0, then we are essentially saying that for each index in
    stick_exact_start (e.g. 150):
        We look for if running contains indices 150-164.
        If it does then index 150 is appended

    Args:
        ind_ref (list) : reference indices
        ind_source (list) : source indices
        range (tuple) : range to append to new list
        prop (float ): proportion (0-1)
    Returns:
        new_ind_ref (list) : indices that fit this criteria
    '''
    new_ind_ref = []
    for ind in ind_ref:
        l = list(np.arange(ind+range[0], ind+range[1]))
        x = list(set(l) & set(ind_source))

        if len(x) / float(len(l)) >= prop:
            new_ind_ref.append(ind)
    return new_ind_ref

def generate_filtered_duration_grid(astro_l, dff_mode=False, behaviour_l='default', filter_duration=(None, None)):
    '''
    Get event grid for specific dff_mode, behaviour list and duration filter
    Args:
        astro_l (list) : list of astrocytes
        dff_mode (bool) : use df/f or not
        behaviour_l (list) : list of behaviours
        filter_duration (tuple) : only events within duration filter
    Returns:
        filtered_grids (list) : list of filtered grids
    '''

    filtered_grids_l = []
    for i, astroA in enumerate(astro_l):
        event_subset = astroA.event_subsets[behaviour_l[i]]
        event_subset_filtered = []

        if filter_duration[0] is None:
            event_subset_min_i = np.copy(event_subset)
        else:
            event_subset_min_i = np.where(astroA.res_d['time_s'][event_subset] > filter_duration[0])[0]

        if filter_duration[1] is None:
            event_subset_max_i = np.copy(event_subset)
        else:
            event_subset_max_i = np.where(astroA.res_d['time_s'][event_subset] < filter_duration[1])[0]

        event_subset_filtered = np.sort(np.array(list(set(list(event_subset_min_i)) & set(list(event_subset_max_i)))))

        if dff_mode:
            filtered_grid = get_dff_grid_from_x2D(astroA.res_d['x2D'][event_subset_filtered], astroA.res_d['dffMax2'][event_subset_filtered], ((astroA.input_shape[0], astroA.input_shape[1])))
        else:
            filtered_grid = get_event_grid_from_x2D(astroA.res_d['x2D'][event_subset_filtered], (astroA.input_shape[0], astroA.input_shape[1])) / len(astroA.indices_d[behaviour_l[i]])

        filtered_grids_l.append(filtered_grid)
    return filtered_grids_l

def split_n_array(arr, n=3):
    '''
    Split array into n equally sized chunks. If not divisible exactly,
    the last chunk is not included
    Args:
        arr : array to split in chunks
        n (int) : number of chunks
    Returns:
        indices_split_l : split array list
    '''
    arr = np.copy(arr)
    indices_split_len = len(arr) // n
    mod_v = (len(arr) % n)
    if mod_v != 0:
        arr = arr[:-mod_v]
        indices_split_len=len(arr) // n
    indices_split_l = [arr[i:i+indices_split_len] for i in range(0, len(arr), indices_split_len)]
    return indices_split_l

def split_n_event_grids(astroA, bh, n=3, minute_frame_scale=True):
    '''
    Split behaviour into n equally size chunks (as many as possible)
    and return event grid

    Args:
        astroA : astrocyte
        bh (str) : behaviour string
        n (int) : number of chunks
        minute_frame_scale (bool) : scale to minute frame

    Returns:
        event_grid_splits (list) : list of grids, split based on parameters
    '''
    inds = astroA.indices_d[bh]
    indices_split_l = split_n_array(inds, n=n)
    indices_d_temp = {'default' : astroA.indices_d['default']}

    for i, ind_split in enumerate(indices_split_l):
        indices_d_temp['split_{}'.format(i)] = ind_split

    event_subsets_temp = get_event_subsets(indices_d_temp, astroA.res_d)
    event_grid_splits = []
    for i in range(len(indices_split_l)):
        event_subset = event_subsets_temp['split_{}'.format(i)]
        event_grid_split = get_event_grid_from_x2D(astroA.res_d['x2D'][event_subset], (astroA.input_shape[0], astroA.input_shape[1]))
        if minute_frame_scale:
            event_grid_split = (event_grid_split / len(indices_split_l[i])) * astroA.minute_frames
        event_grid_splits.append(event_grid_split)
    return event_grid_splits

def get_measure_names(m_l):
    '''
    Return human readable version of measure names

    Args:
        m_l (list) : list of measure names to return name
    Returns:
        mn_l (list) : list of human readable measure names
    '''
    if isinstance(m_l, str):
        m_l = [m_l]

    mn_l = []
    for m in m_l:
        if m == 'area':
            mn_l.append('size')
        elif m == 'dffMax2':
            mn_l.append('amplitude')
        elif m == 'duration':
            mn_l.append('duration')
        elif m == 'time':
            mn_l.append('time')
        elif m == 'time_s' :
            mn_l.append('time (s)')
        else:
            raise NotImplementedError
    if len(mn_l) == 1:
        return mn_l[0]

def speed_event_tuple(astroA, frame_l_l, num_events_only=False, eval_type='mean'):
    '''
    Given list of a list of frames. For each sublist of frames get corresponding
    average speed and events that took place in tuple [avg_speed, event_l]
    Args:
        astroA : the astrocyte
        frame_l_l : list of lists of frames
        num_events_only (bool) :
        eval_type (str) : mean/max
    Returns:
        speed_event_tuple_l (list) : list of form (avg_speed, event_l)
    '''
    if eval_type == 'mean':
        avg_speed_l = [np.mean(astroA.speed_values[split_i]) for split_i in frame_l_l]
    elif eval_type == 'max':
        avg_speed_l = [np.max(astroA.speed_values[split_i]) for split_i in frame_l_l]
    else:
        sys.exit()
    speed_frame_tuple_l = [[avg_speed_l[i], frame_l_l[i]] for i in range(len(frame_l_l))]
    frame_subset_d = {i:speed_frame_tuple_l[i][1] for i in range(len(speed_frame_tuple_l))}
    frame_subset_d['default'] = astroA.indices_d['default']
    event_subset_d = get_event_subsets(frame_subset_d, astroA.res_d, after_i=0, before_i=0, to_print=False)
    if num_events_only:
        speed_event_tuple_l = [[speed_frame_tuple_l[i][0], len(event_subset_d[i])] for i in range(len(speed_frame_tuple_l))]
    else:
        speed_event_tuple_l = [[speed_frame_tuple_l[i][0], event_subset_d[i]] for i in range(len(speed_frame_tuple_l))]
    return speed_event_tuple_l
