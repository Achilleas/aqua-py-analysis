import numpy as np
import math
from analysis.general_utils import aqua_utils, correlation_utils
from preprocessing import analysis_pp
import skimage
import scipy.signal as ss

def get_astro_pair_same_spots_prob(astro_l_pair=None, grid_l=None, p=0.05, dff_mode='False', behaviour='default', move_vector=None):
    '''
    We have j_max pixels same through the astrocyte pairs. Now what is the probability that we get
    P(0 <= x < j_max) pixels in common randomly by selecting 0.05 of the pixels anywhere in each astrocyte?
    Provide either astro_l_pair or grid_l
    '''
    #FILTER top 0.05 of astrocyte list and get how many same pixels in common
    #------------------------------------------------------------------------
    #Filtered top 0.05 non zero grids
    astro_filt_l, astro_all_filt, astro_nz_bool_l, astro_all_nz_bool = get_filters_compare(astro_l_pair=astro_l_pair, grid_l=grid_l,
                                                                                            p=p, behaviour=behaviour, dff_mode=dff_mode, move_vector=move_vector)
    #Non zero sum
    astro_all_nz_sum = np.sum(astro_all_nz_bool)
    #Total joint filtered
    astro_all_filt_sum = np.sum(astro_all_filt > 0)

    #Number of filtered sums
    j_max = astro_all_filt_sum
    common_num = astro_all_nz_sum
    common_pixels = int(p*common_num)

    #We have j_max pixels same through the astrocyte pairs. Now what is the probability that we get
    #P(0 <= x < j_max) pixels in common randomly by selecting 0.05 of the pixels anywhere in each astrocyte?
    s = prob_random_pixels(common_num=common_num,
                           common_pixels=common_pixels,
                           j_max=j_max)

    return (1 - s), astro_filt_l, astro_all_filt, astro_nz_bool_l, astro_all_nz_bool

def get_filters_compare(astro_l_pair=None, grid_l=None, p=0.05, dff_mode=False, behaviour='default', move_vector=None):
    '''
    Return filtered grid based on parameters
    Providing grid_a and grid_b overrides astro_l_pair, dff_mode and behaviour.

    Args:
        astro_l_pair (list) : list of astrocytes (can be more than a pair)
        grid_l (list) : list of grids (overrides astro_l_pair for filtering)
        p (float) : top % values to include only. p=1 include all values
        dff_mode (bool) : df/f boolean
        behaviour (str) : the behaviour string
        move_vector (tuple) : apply move vector (translation)
    Returns:
        astro_filt_l (list) : filtered list of grids
        astro_all_filt (list) : product of all filtered grids
        astro_nz_bool_l (list) : filtered list of grids (boolean vaues)
        astro_all_nz_bool (list) : product of all non zero filtered grids
    '''
    if (astro_l_pair is None) and (grid_l is None):
        print('get_filters_compare, provided astro_l_pair or grid_l, both are None')
        return None
    #Filtered top 0.05 non zero grids
    if isinstance(behaviour, str):
        behaviour = [behaviour, behaviour]

    if (grid_l is not None):
        astro_filt_l = [get_filtered_grid(grid, threshold=p) for grid in grid_l]
    else:
        if dff_mode:
            astro_filt_l = [get_filtered_grid(A.event_grids_dff[behaviour[i]], threshold=p) for i, A in enumerate(astro_l_pair)]
        else:
            astro_filt_l = [get_filtered_grid(A.event_grids_event_norm[behaviour[i]], threshold=p) for i, A in enumerate(astro_l_pair)]

    if move_vector is not None:
        astro_filt_l[1] = move_array_2d(astro_filt_l[1], move_vector)

    #Filtered joint
    astro_all_filt = np.copy(astro_filt_l[0])
    for i in range(1, len(astro_filt_l)):
        astro_all_filt *= astro_filt_l[i]

    #GET non zero common in astrocyte list
    #Non zero grids
    astro_nz_bool_l = [(A.event_grids[behaviour[i]] > 0) for i, A in enumerate(astro_l_pair)]
    #Total non zero values in each grid
    astro_nz_len_l = [np.sum(A_nz) for A_nz in astro_nz_bool_l]
    #Non zero bools grid
    astro_all_nz_bool = np.copy(astro_nz_bool_l[0])
    for i in range(1, len(astro_nz_bool_l)):
        astro_all_nz_bool *= astro_nz_bool_l[i]
    astro_all_nz_sum = np.sum(astro_all_nz_bool)

    return astro_filt_l, astro_all_filt, astro_nz_bool_l, astro_all_nz_bool

def get_filters_compare_from_grids(grid_l_pair, p=0.05):
    '''
    Filter compare from grids. Same as function above but only for grids
    '''
    #Filtered top 0.05 non zero grids
    astro_filt_l = [get_filtered_grid(grid_l_pair[i], threshold=p) for i in range(len(grid_l_pair))]
    #Filtered joint
    astro_all_filt = np.copy(astro_filt_l[0])
    for i in range(1, len(astro_filt_l)):
        astro_all_filt *= astro_filt_l[i]

    #GET non zero common in astrocyte list
    #Non zero grids
    astro_nz_bool_l = [(grid_l_pair[i] > 0) for i in range(len(grid_l_pair))]
    #Total non zero values in each grid
    astro_nz_len_l = [np.sum(A_nz) for A_nz in astro_nz_bool_l]
    #Non zero bools grid
    astro_all_nz_bool = np.copy(astro_nz_bool_l[0])
    for i in range(1, len(astro_nz_bool_l)):
        astro_all_nz_bool *= astro_nz_bool_l[i]
    astro_all_nz_sum = np.sum(astro_all_nz_bool)

    return astro_filt_l, astro_all_filt, astro_nz_bool_l, astro_all_nz_bool

def get_filtered_grid(grid, threshold=0.1):
    '''
    Args:
        grid : the grid to apply threshold filter
        threshold (float) : for 0.5 set lowest 50% of values to 0
    Returns:
        grid_cp : thresholded grid
    '''
    arr_s = np.sort(grid.flatten())
    arr_s = arr_s[arr_s != 0]
    i_threshold = int(len(arr_s) * threshold)

    grid_cp = np.copy(grid)
    grid_cp[grid_cp <= arr_s[-i_threshold]] = 0
    return grid_cp

def prob_random_pixels(common_num, common_pixels, j_max):
    '''
    Helper function, combinatorial probability
    '''
    s = 0
    for j in range(j_max):
        s += combinatorial_prob_common(N=common_num, m=common_pixels, j=j)
        print(s)
    return s

def combinatorial_prob_common(N, m, j):
    '''
    Given N possible values (e.g. N pixels)
    m of those values are chosen (without replacement)

    What is the probability that if I were to choose m values again (no replacement)
    that j number of them would be the same
    '''
    #return (comb(m, j) * comb(N-m, m-j)) / comb(N, m)
    v_up = ((math.factorial(m)**2)* math.factorial(N-m)**2)
    v_down = ((math.factorial(j) * math.factorial(m-j)**2 * math.factorial(N) * math.factorial(N+j-2*m)))
    return v_up/v_down

def move_array_2d(arr, move_vector):
    '''
    Args:
        arr (np.ndarray) : the array
        move_vector (tuple) : move vector to apply to array
    Returns:
        arr_shifted (np.ndarray) : shifted array
    '''
    if move_vector is None:
        return arr
    arr_shifted = np.copy(arr)

    arr_shifted = np.roll(arr_shifted, shift=move_vector[0], axis=0)
    arr_shifted = np.roll(arr_shifted, shift=move_vector[1], axis=1)

    if move_vector[0] >= 0:
        arr_shifted[:move_vector[0], :] = 0
    else:
        arr_shifted[move_vector[0]:, :] = 0

    if move_vector[1] >= 0:
        arr_shifted[:, :move_vector[1]] = 0
    else:
        arr_shifted[:, move_vector[1]:] = 0

    return arr_shifted

def get_fake_astrocyte_sample_v2(astro, astro_grid, mode='bool', filter_ratio=0.9):
    '''
    (Deprecated: see get_fake_astrocyte_sample_from_areas)
    Generates a fake astrocyte sample from astrocyte grid. Takes disconnected
    components from astro_grid, recording their distribution of sizes. Then we
    sample random ellipses (varying shape) until same number of pixels as total
    area of astro grid is covered.
    '''
    labeled, ncomponents, sizes_l = analysis_pp.get_component_info(astro_grid)

    sizes_l_s = np.sort(sizes_l)
    max_pixels = np.sum(sizes_l)

    if len(sizes_l_s) < 10:
        min_ind, max_ind = 0, len(sizes_l_s)
    else:
        min_ind, max_ind = 0, int((len(sizes_l_s)*filter_ratio))

    border = astro.res_d['border_mask']
    num_valid = int(np.sum(border))

    filled = np.zeros(border.shape)

    if len(sizes_l_s) == 0:
        return filled

    curr_pixels = 0
    while curr_pixels < max_pixels:
        #Get area of ellipse based on sizes
        curr_area = sizes_l_s[np.random.randint(min_ind, max_ind)]
        #Get coordinate center
        y_coord_center, x_coord_center = np.transpose(np.nonzero(border))[np.random.randint(num_valid)]

        dv = np.random.uniform(low=0.5, high=1.5)
        r_radius = np.sqrt(curr_area/np.pi)/dv
        c_radius = np.sqrt(curr_area/np.pi)/(1/dv)

        rr, cc = skimage.draw.ellipse(r=y_coord_center, c=x_coord_center,
                              r_radius=r_radius, c_radius=c_radius,
                              shape=filled.shape, rotation=np.random.randint(-90, 90))
        if mode == 'bool':
            filled[rr, cc] = 1
        elif mode == 'append':
            filled[rr, cc] += 1
        curr_pixels += len(rr)
    print('max pixels: {} curr pixels: {}'.format(max_pixels, curr_pixels))
    return filled

def get_fake_astrocyte_sample_from_areas(astro, sizes_l, mode='append', filter_ratio=0.9, return_info=False, border_grid=None):
    '''
    Get fake astrocyte sample from areas.

    Args:
        astro : the astrocyte
        sizes_l (list) : list of sizes
        mode (str) : 'bool'/'append'
        filter_ratio (float) : filter ratio
        return_info (bool) : return center and radius lists in addition
        border_grid : border grid to filter events
    '''
    sizes_l_s = np.sort(sizes_l)
    max_pixels = np.sum(sizes_l)
    min_ind, max_ind = 0, int((len(sizes_l_s)*filter_ratio))

    if border_grid is not None:
        border = border_grid
    else:
        border = astro.res_d['border_mask']

    num_valid = int(np.sum(border))

    filled = np.zeros(border.shape)
    curr_pixels = 0
    center_l = []
    radius_l = []
    while curr_pixels < max_pixels:
        #Get area of ellipse based on sizes
        curr_area = sizes_l_s[np.random.randint(min_ind, max_ind)]
        #Get coordinate center
        y_coord_center, x_coord_center = np.transpose(np.nonzero(border))[np.random.randint(num_valid)]

        dv = np.random.uniform(low=0.5, high=1.5)
        r_radius = np.sqrt(curr_area/np.pi)/dv
        c_radius = np.sqrt(curr_area/np.pi)/(1/dv)

        rr, cc = skimage.draw.ellipse(r=y_coord_center, c=x_coord_center,
                              r_radius=r_radius, c_radius=c_radius,
                              shape=filled.shape, rotation=np.random.randint(-90, 90))
        if mode == 'bool':
            filled[rr, cc] = 1
        elif mode == 'append':
            filled[rr, cc] += 1
        curr_pixels += len(rr)

        center_l.append((y_coord_center, x_coord_center))
        radius_l.append((r_radius, c_radius))
    print('max pixels: {} curr pixels: {}'.format(max_pixels, curr_pixels))

    if return_info:
        return {'sample' : filled, 'center_l' : center_l, 'radius_l' : radius_l}
    return filled

def split_astro_grid(astro, split_frames=1000, bk='default', inds_subset=None):
    '''
    Split astrocyte event grid per set of X frames
    Args:
        astro : Astrocyte
        split_frames (int) : number of frames for each split
        bk (str) : behaviour
        inds_subset (dict) : dictionary
    '''
    event_grid_splits_l = []

    if inds_subset is None:
        all_inds = astro.indices_d[bk]
        all_inds = np.sort(all_inds)
    else:
        all_inds = inds_subset
        all_inds = np.sort(inds_subset)

    inds_split = {'split_{}'.format(i) : all_inds[split_frames*(i):(split_frames*(i+1))] for i in range(len(all_inds) // split_frames)}
    inds_split['default'] = all_inds
    event_subset_split = aqua_utils.get_event_subsets(inds_split, astro.res_d, after_i=0, before_i=0, to_print=False)
    for i in range(len(all_inds) // split_frames):
        event_subset_split_i = event_subset_split['split_{}'.format(i)]
        event_grid_split_i = aqua_utils.get_event_grid_from_x2D(astro.res_d['x2D'][event_subset_split_i], (astro.input_shape[0], astro.input_shape[1])) / len(event_subset_split_i)
        event_grid_splits_l.append(event_grid_split_i)
    return event_grid_splits_l

def get_align_translation_clandmark(astro_target, astro_source):
    '''
        Given astrocyte target and astrocyte, return (x, y), the translation in x, y to be applied to astrocyte
        to get to astrocyte target. We use clandmark center to obtain this translation
    '''
    c_target = astro_target.res_d['clandmark_center']
    c_source = astro_source.res_d['clandmark_center']

    res = c_target - c_source
    res = [int(res[0]), int(res[1])]
    return res

def get_align_translation_xcorr(astro_target, astro_source):
    '''
    '''
    border_target = astro_target.res_d['border_mask'] - 0.5
    border_source = astro_source.res_d['border_mask'] - 0.5
    corr_res, _, move_vector, _ = correlation_utils.get_cross_correlation_2D_info_compare(border_target, border_source, normalize=False)
    return move_vector

def alignment_counter(astro_target, astro_source, grid_target=None, grid_source=None, n_fake_samples=0,
                            align_setting='xcorr', eval_setting='counter', fake_sample_setting='from_grid',
                            move_vector=None, p=1, dff_mode=False, behaviour='default', filter_duration=(None, None),
                            with_output_details=False, border_nan=True, target_border_grid=None, source_border_grid=None):
    '''
    Given astro target and astro source and probability p.
    Find top p values in astro_target and astro_source
    Align source to target based on their borders.
    Calculate how many pixels in common with these top p values
    Apply these same for fake samples

    Returns list of num values in common

    grids may be provided instead of using p and astro

    eval_setting:
        counter : count number of common spots after alignment
        xcorr   : normalized 2d correlation value at alingment position
    fake_sample_setting:
        from_grid  : create fake sample from grid, splitting the grid into different disjoint
                    components and using the sizes of this to create the fake samples
        from_astro : use information from the astrocyte to create the fake samples (DO NOT USED UNLESS P=1)
    filter_duration:
        Filter the minimum and maximum duration of events allowed to be used (must insert astro_target and astro_source)

    Commonly, use {eval_setting=counter,fake_sample_setting=from_grid, p=p} to get top p values and compare counts
             or use {eval_setting=xcorr,fake_sample_setting=from_astro,p=1} to get xcorr values to compare whole astro

    '''
    if isinstance(behaviour, str):
        behaviour = [behaviour, behaviour]

    output_details = {}
    print('WHAT IS BEHAVIOR:', behaviour, type(behaviour))
    if (behaviour[0] not in astro_source.indices_d.keys()) or (behaviour[1] not in astro_source.indices_d.keys()):
        print('Behaviour : {} not in astro_source, (alignment counter)'.format(behaviour))
    astro_l_pair = [astro_target, astro_source]

    #If we are filtering duration: we generate filtered grids and use those
    if (filter_duration[0] is not None) or (filter_duration[1] is not None):
        print('Generating filtered duration grid...')
        grid_target, grid_source = aqua_utils.generate_filtered_duration_grid([astro_target, astro_source], dff_mode=dff_mode,
                                                                        behaviour_l=behaviour, filter_duration=filter_duration)

    if (grid_target is None) or (grid_source is None):
        print('Get filters compare,', behaviour, dff_mode)
        astro_filt_l, _, _,_ = get_filters_compare(astro_l_pair, p=p, dff_mode=dff_mode, behaviour=behaviour)
    else:
        astro_filt_l,_,_,_ = get_filters_compare_from_grids([grid_target, grid_source], p=p)

    #Then find the xcorr and clandmark
    if align_setting == 'xcorr':
        move_vector = get_align_translation_xcorr(astro_target, astro_source)
    elif align_setting == 'clandmark':
        move_vector = get_align_translation_clandmark(astro_target, astro_source)
    elif align_setting == 'param':
        pass
    else:
        print('Invalid align setting')
        return None
    astro_filt_bool_l = [astro_filt_l[0] != 0, astro_filt_l[1] != 0]

    #Intersection of the 2 astrocyte borders if we are using borders for normalization (for corr)
    fake_sample_border_grid=None
    if border_nan:
        if (target_border_grid is not None) and (source_border_grid is not None):
            source_border = source_border_grid
            target_border = target_border_grid
            fake_sample_border_grid = source_border_grid
        else:
            source_border = astro_source.border
            target_border = astro_target.border

        source_border = source_border if move_vector is None else move_array_2d(source_border, move_vector)
        border_join = target_border * source_border
        output_details['border_join'] = border_join
        output_details['border_1'] = target_border
        output_details['border_2'] = source_border

    if eval_setting == 'counter':
        num_self = np.sum(astro_filt_bool_l[0])
        num_compare = np.sum(astro_filt_bool_l[0] * move_array_2d(astro_filt_bool_l[1], move_vector))
    elif eval_setting == 'xcorr' or eval_setting == 'xcorr_random_both':
        print('EVAL SETTING XCORR')
        grid_1 = astro_filt_l[0]
        grid_2 = move_array_2d(astro_filt_l[1], move_vector)

        grid_1_norm = correlation_utils.normalize_grid(grid_1, border=border_join if border_nan else None)
        grid_2_norm = correlation_utils.normalize_grid(grid_2, border=border_join if border_nan else None)

        output_details['grid_1'] = grid_1
        output_details['grid_2'] = grid_2

        output_details['grid_1_norm'] = grid_1_norm
        output_details['grid_2_norm'] = grid_2_norm

        norm_val = np.sum(border_join) if border_nan else (grid_1_norm.shape[0]*grid_1_norm.shape[1])

        corr_res = np.sum(grid_1_norm * grid_2_norm) / norm_val
        num_compare = corr_res
        num_self = np.sum(grid_1_norm * grid_1_norm) / norm_val
        print('DONE EVAL XCORR')
    #Find maximum xcorr, allowing movement of the grid instead of providing a move vector and fixing
    elif eval_setting == 'xcorr_free':
        grid_1 = astro_filt_l[0]
        grid_2 = astro_filt_l[1]

        grid_1_norm = correlation_utils.normalize_grid(grid_1, border=border_join if border_nan else None)
        grid_2_norm = correlation_utils.normalize_grid(grid_2, border=border_join if border_nan else None)

        output_details['grid_1'] = grid_1
        output_details['grid_2'] = grid_2

        output_details['grid_1_norm'] = grid_1_norm
        output_details['grid_2_norm'] = grid_2_norm

        corr_matrix = ss.correlate2d(grid_1_norm, grid_2_norm, mode='full', boundary='fill', fillvalue=0)
        #Both grid 1 and grid 2 must be same dims
        norm_val = np.sum(border_join) if border_nan else (grid_1_norm.shape[0]*grid_1_norm.shape[1])
        corr_matrix = corr_matrix / norm_val
        corr_res = np.max(corr_matrix)

        best_arg = np.unravel_index(np.argmax(corr_matrix), corr_matrix.shape)
        move_vector_free = best_arg - np.array([corr_matrix.shape[0] // 2, corr_matrix.shape[1] // 2])
        num_compare = corr_res

        output_details['move_vector_free'] = move_vector_free
        output_details['corr_matrix'] = corr_matrix
        #It's mathematically 1...
        num_self = 1
        #'grid_1' : grid_1_norm, 'grid_2' : grid_2_norm

    if eval_setting == 'counter':
        fake_sample_mode = 'bool'
    else:
        fake_sample_mode = 'append'

    print('FAKE SAMPLE MODE:', fake_sample_mode)
    num_fake = []

    output_details['fake_samples'] = []
    output_details['fake_move_vector_free_l'] = []
    output_details['fake_corr_matrix_l'] = []

    #nfake samples
    for i in range(n_fake_samples):
        if fake_sample_setting == 'from_grid':
            sample_i = get_fake_astrocyte_sample_v2(astro=astro_source, astro_grid=astro_filt_bool_l[1], mode=fake_sample_mode)
        elif fake_sample_setting == 'from_astro':
            event_areas = astro_source.res_d['area'][astro_source.event_subsets[behaviour[1]]] / (astro_source.spatial_res**2)
            sample_i = get_fake_astrocyte_sample_from_areas(astro_source, sizes_l=event_areas, mode=fake_sample_mode, filter_ratio=1, return_info=False, border_grid=fake_sample_border_grid)

        sample_i = move_array_2d(sample_i, move_vector)

        if eval_setting == 'counter':
            num_fake.append(np.sum(astro_filt_bool_l[0] * sample_i))
        elif eval_setting == 'xcorr':
            print('Eval setting xcorr...')

            grid_1_norm = correlation_utils.normalize_grid(astro_filt_l[0], border=border_join if border_nan else None)
            grid_2_norm = correlation_utils.normalize_grid(sample_i, border=border_join if border_nan else None)

            corr_res = ss.correlate2d(grid_1_norm, grid_2_norm, mode='valid', boundary='fill', fillvalue=0)[0,0]
            norm_val = np.sum(border_join) if border_nan else (grid_1_norm.shape[0]*grid_1_norm.shape[1])
            corr_res = corr_res / norm_val
            num_fake.append(corr_res)
            print('ADDED fake {}'.format(i))
        elif eval_setting == 'xcorr_free':
            print('Eval setting xcorr_free...')

            grid_1_norm = correlation_utils.normalize_grid(astro_filt_l[0], border=border_join if border_nan else None)
            grid_2_norm = correlation_utils.normalize_grid(sample_i, border=border_join if border_nan else None)

            min_value = min(np.min(grid_1_norm), np.min(grid_2_norm))
            fake_corr_matrix = ss.correlate2d(grid_1_norm, grid_2_norm, mode='full', boundary='fill', fillvalue=0)
            norm_val = np.sum(border_join) if border_nan else (grid_1_norm.shape[0]*grid_1_norm.shape[1])

            fake_corr_matrix = fake_corr_matrix / norm_val
            fake_corr_res = np.max(fake_corr_matrix)

            best_arg = np.unravel_index(np.argmax(fake_corr_matrix), fake_corr_matrix.shape)
            move_vector_free = best_arg - np.array([fake_corr_matrix.shape[0] // 2, fake_corr_matrix.shape[1] // 2])

            num_fake.append(fake_corr_res)
            output_details['fake_move_vector_free_l'].append(move_vector_free)
            output_details['fake_corr_matrix_l'].append(fake_corr_matrix)

            print('ADDED fake {}'.format(i))
        elif eval_setting == 'xcorr_random_both':
            print('GENERATING FAKE SAMPLES')
            event_areas_source = astro_source.res_d['area'][astro_source.event_subsets[behaviour[1]]] / (astro_source.spatial_res**2)
            sample_source_i = get_fake_astrocyte_sample_from_areas(astro_source, sizes_l=event_areas_source, mode=fake_sample_mode, filter_ratio=1, return_info=False, border_grid=fake_sample_border_grid)

            event_areas_target = astro_target.res_d['area'][astro_target.event_subsets[behaviour[1]]] / (astro_target.spatial_res**2)
            sample_target_i = get_fake_astrocyte_sample_from_areas(astro_target, sizes_l=event_areas_target, mode=fake_sample_mode, filter_ratio=1, return_info=False, border_grid=fake_sample_border_grid)

            sample_target_i = move_array_2d(sample_target_i, move_vector)

            grid_1_norm = correlation_utils.normalize_grid(sample_source_i, border=border_join if border_nan else None)
            grid_2_norm = correlation_utils.normalize_grid(sample_target_i, border=border_join if border_nan else None)

            corr_res = ss.correlate2d(grid_1_norm, grid_2_norm, mode='valid', boundary='fill', fillvalue=0)[0,0]
            norm_val = np.sum(border_join) if border_nan else (grid_1_norm.shape[0]*grid_1_norm.shape[1])
            corr_res = corr_res / norm_val
            num_fake.append(corr_res)
            print('ADDED fake {}'.format(i))
        output_details['fake_samples'].append(sample_i)

    output_details['eval_setting'] = eval_setting
    output_details['num_self'] = num_self
    output_details['num_compare'] = num_compare
    output_details['num_fake'] = num_fake
    output_details['move_vector'] = move_vector

    if with_output_details:
        return output_details
    else:
        return {'num_self' : num_self, 'num_compare' : num_compare,
                'num_fake' : num_fake, 'move_vector' : move_vector}

def get_move_vector_xcorr_default(astro_a, astro_b):
    '''
    Obtain move vector by aligning borders of the 2 astrocytes

    Args:
        astro_a : astrocyte
        astro_b : astrocyte
    Returns:
        move_vector : the move vector
    '''
    d_temp = alignment_counter(astro_a, astro_b,
                                n_fake_samples=0,
                                align_setting='xcorr',
                                eval_setting='xcorr',
                                fake_sample_setting='from_astro',
                                p=1,
                                behaviour='default')
    move_vector = d_temp['move_vector']
    return move_vector
