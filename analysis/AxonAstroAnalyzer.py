import numpy as np
import os, sys, glob
import scipy
import scipy.io
from preprocessing import analysis_pp
from analysis.general_utils import aqua_utils, saving_utils
import copy

class AxonAstroAnalyzer():
    def __init__(self, base_path, day, aqua_bound=False, with_landmarks=True, num_axons=3, num_astro_far=3):
        self.base_path = base_path
        self.day = day
        self.aqua_bound = aqua_bound
        self.num_axons = num_axons
        self.num_astro_far = num_astro_far
        self.with_landmarks = with_landmarks

        self.experiment_path = os.path.join(self.base_path, 'day_' + str(self.day))
        self.behaviour_base_path = 'roi_data_merged/other.csv'
        self.oscilloscope_base_path = 'oscilloscope/oscilloscope.csv'
        self.landmarks_base_path = 'axon_astro_landmarks/'

        self.delay_min = 0
        self.delay_max = 30

        #Axon fix area
        self.pass_area_rate = 0.2

        if aqua_bound:
            self.aqua_base_path = 'aqua_bound/aqua_merged/merged.pkl'
        else:
            self.aqua_base_path = 'aqua_no_bound/aqua_merged/merged.pkl'

        self.behaviours_path = os.path.join(self.experiment_path, self.behaviour_base_path)
        self.oscilloscope_path = os.path.join(self.experiment_path, self.oscilloscope_base_path)
        self.save_res_path = os.path.join(self.experiment_path, self.aqua_base_path)

        if with_landmarks:
            self.landmarks_path = os.path.join(self.experiment_path, self.landmarks_base_path)

        ###################################################
        # Behavioural indices
        self.indices_d, _, l = analysis_pp.read_behaviour_indices_no_whiskers_axon_astro(self.behaviours_path, self.oscilloscope_path)
        self.stick_bin, self.speed_bin, self.pupil_values, self.speed_values = l

        ###################################################
        print(self.save_res_path)
        print(os.path.exists(self.save_res_path))
        self.res_d, self.input_shape_l, self.fr_inv, self.spatial_res = saving_utils.load_pickle(self.save_res_path)
        self.res_d = aqua_utils.apply_res_d_filter(self.res_d, min_event_duration=2, min_um_size=2)
        self.fr = 1.0 /self.fr_inv
        self.res_d['time'] = self.res_d['tEnd'] - self.res_d['tBegin']
        self.res_d['time_s'] = self.res_d['time'] / self.fr

        self.minute_frames = int(np.round(self.fr * 60))
        self.input_shape = self.input_shape_l[0]
        print(self.input_shape_l, self.fr_inv, self.spatial_res)
        print(self.res_d.keys())

        self.total_indices = len(self.indices_d['default'])
        self.total_events = len(self.res_d['tBegin'])
        ###################################################

        self.bright_spot_events_l = []
        self.ignore_events_l = []

        #Axon, astro ids
        print('Setting strs...')
        self.setup_strs()

        #Axon only bound
        print('Setting axon only bound...')
        self.setup_axon_only_bound()

        #Landmarks
        print('Setting landmarks...')
        self.setup_landmarks()

        #Event subsets
        print('Setting subsets...')
        self.setup_event_subsets()

        #Centroids
        print('Setting up centroids...')
        self.setup_centroids()

        #Event grids
        print('Event grids...')
        self.setup_grids() #(self.event_grids, self.event_grids_norm_rel)
        #Activity ratio
        print('Activity ratios...')
        self.setup_activity_ratios()

        #Events and corresponding times for each behavioural state for each landmark
        print('Setup contained...')
        self.setup_contained()

        #Find number of events for each behavioural state for each landmark with delay
        print('Setting num events...')
        self.setup_num_events()

        #Find number of events that take place after axon_x for each relevant location within 5 seconds
        print('Setting ddr heatmaps...')
        self.setup_ddr_heatmaps()

        self.id = os.path.join(os.path.basename(os.path.normpath(self.base_path)), 'day_' + str(self.day))
        self.print_id = '_'.join(self.id.split('/'))

    def setup_event_subsets(self):
        self.event_subsets = aqua_utils.get_event_subsets(self.indices_d, self.res_d, after_i=0, before_i=0, to_print=True)
        self.event_subsets_old = np.copy(self.event_subsets)
        keys_to_del = []

        for k in self.event_subsets.keys():
            self.event_subsets[k] = np.sort(list(set(self.event_subsets[k]) - set(self.bright_spot_events_l) - set(self.ignore_events_l)))

            if len(self.event_subsets[k]) == 0:
                keys_to_del.append(k)

        for k in keys_to_del:
            del self.event_subsets[k]

    def setup_landmarks(self):
        self.landmark_borders = {}
        self.landmark_line_borders = {}
        self.landmark_line_locs = {}
        self.landmark_locs = {}
        for filepath in glob.glob(os.path.join(self.landmarks_path, '*.mat')):
            print(filepath)
            mat = scipy.io.loadmat(filepath)
            name = os.path.basename(filepath).split('.mat')[0]

            self.landmark_borders[name] = np.zeros([self.input_shape[0], self.input_shape[1]])
            self.landmark_line_borders[name] = np.zeros([self.input_shape[0], self.input_shape[1]])

            landmark_locs_np = None
            landmark_line_locs_np = None

            for i in range(len(mat['bd0'][0])):
                landmark_locs_i = np.unravel_index(np.array(mat['bd0'][0][i][0][1]), [self.input_shape[0], self.input_shape[1]], order='F')

                if i == 0:
                    landmark_locs_np = np.array(landmark_locs_i)
                else:
                    landmark_locs_np = np.concatenate([landmark_locs_np, np.array(landmark_locs_i)], axis=1)

                landmark_line_locs_i = [[], []]

                line_locs = np.array(mat['bd0'][0][i][0][0][0][0])
                for line_loc in line_locs:
                    landmark_line_locs_i[0].append(line_loc[0])
                    landmark_line_locs_i[1].append(line_loc[1])
                landmark_line_locs_i[0] = np.array(landmark_line_locs_i[0])
                landmark_line_locs_i[1] = np.array(landmark_line_locs_i[1])
                landmark_line_locs_i = tuple(landmark_line_locs_i)

                if i == 0:
                    landmark_line_locs_np = np.array(landmark_line_locs_i)
                else:
                    landmark_line_locs_np = np.concatenate([landmark_line_locs_np, np.array(landmark_line_locs_i)], axis=1)

            self.landmark_locs[name]= [[], []]
            self.landmark_line_locs[name] = [[], []]

            self.landmark_locs[name][0] = tuple((landmark_locs_np[0, :].flatten()).tolist())
            self.landmark_locs[name][1] = tuple((landmark_locs_np[1, :].flatten()).tolist())

            self.landmark_line_locs[name][0] = tuple((landmark_line_locs_np[0, :].flatten()-1).tolist())
            self.landmark_line_locs[name][1] = tuple((landmark_line_locs_np[1, :].flatten()-1).tolist())

            self.landmark_locs[name] = tuple(self.landmark_locs[name])
            self.landmark_line_locs[name] = tuple(self.landmark_line_locs[name])
            self.landmark_borders[name][self.landmark_locs[name]] = 1

            self.landmark_line_borders[name][self.landmark_line_locs[name]] = 1

    def setup_axon_only_bound(self):
        self.axon_only_bound = None
        self.axon_only_path = os.path.join(self.experiment_path, 'axon_only_bound', 'axon_bound.mat')

        if os.path.exists(self.axon_only_path):
            self.axon_only_bound = np.zeros([self.input_shape[0], self.input_shape[1]])

            mat = scipy.io.loadmat(self.axon_only_path)
            for i in range(len(mat['bd0'][0])):
                bound_locs = np.unravel_index(np.array(mat['bd0'][0][i][0][1])-1, [self.input_shape[0], self.input_shape[1]], order='F')
                self.axon_only_bound[bound_locs] = 1
        else:
            print('Axon only bound not found... self.axon_only_bound set to None')

    def setup_centroids(self):
        self.axon_centroids = {}
        self.event_centroids = {}
        self.landmark_centroids = {}

        for axon_str in self.axon_strs:
            self.axon_centroids[axon_str] = (int(np.floor(np.mean(self.landmark_locs[axon_str][0]))), int(np.floor(np.mean(self.landmark_locs[axon_str][1]))))

        for landmark_str in self.landmark_locs.keys():
            self.landmark_centroids[landmark_str] = (int(np.floor(np.mean(self.landmark_locs[landmark_str][0]))), int(np.floor(np.mean(self.landmark_locs[landmark_str][1]))))

        for bk in self.indices_d.keys():
            self.event_centroids[bk] = aqua_utils.get_event_centroids_from_x2D(self.res_d['x2D'][self.event_subsets[bk]], (self.input_shape[0], self.input_shape[1]))

    def setup_strs(self):
        #axons
        self.axon_strs = ['axon_{}'.format(i+1) for i in range(self.num_axons)]
        self.axon_keys = []
        self.compare_tuples = {}

        for k in self.axon_strs:
            axon_num = int(k.split('_')[-1])

            self.compare_tuples[k] = {}
            self.compare_tuples[k] = {'axon' : k,
                                 'axon_far' : 'axon_{}_far'.format(axon_num),
                                 'close_1'  : 'astro_{}_close_1'.format(axon_num),
                                 'close_2'  : 'astro_{}_close_2'.format(axon_num)}
            for af_i in range(self.num_astro_far):
                self.compare_tuples[k]['far_' + str(af_i + 1)] = 'astro_far_' + str(af_i + 1)

        for k in self.axon_strs:
            self.axon_keys.append(k)
            self.axon_keys.append(k + '_far')

        print(self.compare_tuples)

    def setup_contained(self):
        #Events contained for each behavioural state for each landmark id
        self.events_contained = {}
        self.events_excluded = {}
        #Find times of events contained for each behavioural state for each landmark id
        self.times_contained = {}
        self.astro_landmark_bound_events_contained = {}

        #Setup axon bound events (outside of astrocyte)
        if self.axon_only_bound is not None:
            self.axon_bound_events_contained = {}
            self.axon_bound_events_excluded = {}
            self.axon_bound_times_contained = {}

        for bk in self.indices_d:
            self.events_contained[bk], self.events_excluded[bk] = self.get_events_contained(self.landmark_borders, self.event_centroids[bk], self.event_subsets[bk])
            self.times_contained[bk] = self.get_times_contained(self.landmark_borders, self.events_contained[bk], self.event_centroids[bk])

            if self.axon_only_bound is not None:
                ax_ev_c = self.get_events_contained({'axon_only_bound' : self.axon_only_bound}, self.event_centroids[bk], self.event_subsets[bk])
                self.axon_bound_events_contained[bk], self.axon_bound_events_excluded[bk] = ax_ev_c[0]['axon_only_bound'], ax_ev_c[1]['axon_only_bound']
                self.axon_bound_times_contained[bk] = self.get_times_contained({'axon_only_bound' : self.axon_only_bound}, ax_ev_c[0], self.event_centroids[bk])['axon_only_bound']

        #Setup events ONLY in astrocyte excluding the axon events (doesn't include all astrocyte events)
        for bk in self.indices_d:
            self.astro_landmark_bound_events_contained[bk] = []
            for k in self.events_contained[bk].keys():
                if 'astro_' in k:
                    self.astro_landmark_bound_events_contained[bk].extend(self.events_contained[bk][k])
            self.astro_landmark_bound_events_contained[bk] = np.sort(np.array(list(set(self.astro_landmark_bound_events_contained[bk]))))

    def setup_num_events(self):
        self.num_events = {}
        self.num_events_norm = {}
        for bk in self.indices_d:
            self.num_events[bk], self.num_events_norm[bk] = self.get_num_events(self.compare_tuples, self.times_contained[bk])

    def setup_ddr_heatmaps(self):
        self.axon_delays = {}
        #self.coords_from_center = {}
        self.ddr_heatmaps = {}
        self.ddr_heatmaps_count = {}

        for bk in self.indices_d.keys():
            #print('------------------')
            #print(bk)
            self.axon_delays[bk] = {}
            #coords_from_center[bk] = {}
            self.ddr_heatmaps_count[bk] = {}
            self.ddr_heatmaps[bk] = {}

            for axon_str in self.axon_strs:
                print(axon_str)
                print('\n')
                #----------------------------------------------------------------------
                #Beginning times of events in bk behaviour
                event_times = self.res_d['tBegin'][self.event_subsets[bk]]
                #Beginning times of events in bk behaviour inside axon only
                axon_times = np.array([times[0] for times in self.times_contained[bk][axon_str]], dtype=int)

                #coords_from_center[bk][axon_str] = [(np.array(event_centroid) - np.array(axon_centroids[axon_str])) for event_centroid in event_centroids[bk]]
                #Get delays of all behaviour events after axon
                self.axon_delays[bk][axon_str] = self.get_event_axon_delays(event_times, axon_times, self.delay_min, self.delay_max)

                #----------------------------------------------------------------------
                #Get heatmaps corresponding to these delays
                self.ddr_heatmaps[bk][axon_str], self.ddr_heatmaps_count[bk][axon_str] = self.get_ddr_heatmaps(self.event_centroids[bk],
                                                                                                        self.event_subsets[bk],
                                                                                                        self.axon_delays[bk][axon_str])
                #delay_landmark_ids, delay_means, delay_stds = self.get_values_heatmap_from_landmarks(self.ddr_heatmaps[bk][axon_str],
                #                                                                             self.ddr_heatmaps_count[bk][axon_str],
                #                                                                             self.landmark_locs)
                #for i in range(len(delay_landmark_ids)):
                #    print('{:<10}\t{:>10}{:.2f}\tstd\t{:.2f}'.format(delay_landmark_ids[i], 'delay: mean ', delay_means[i], delay_stds[i]))

    def get_events_contained(self, landmark_borders, event_centroids, event_indices):
        events_contained = {lk: [] for lk in landmark_borders.keys()}
        events_excluded = {lk: [] for lk in landmark_borders.keys()}
        #Contains list of event indices contained
        for lk in landmark_borders.keys():
            for event_centroid, event_i in zip(event_centroids, event_indices):
                if landmark_borders[lk][int(event_centroid[0]), int(event_centroid[1])] == 1:
                    #Remove any events in axons that are <20% of their area for (axons only)
                    #This is very typically noise of the blobs due to z axis and movement shifts
                    '''
                    if lk in self.axon_keys:
                        min_area = np.sum(landmark_borders[lk]) * self.spatial_res**2 * self.pass_area_rate
                        if min_area > self.res_d['area'][event_i]:
                            events_excluded[lk].append(event_i)
                            continue
                    '''
                    events_contained[lk].append(event_i)

        return events_contained, events_excluded

    def get_times_contained(self, landmark_borders, events_contained, event_centroids):
        times_contained = {lk: [] for lk in landmark_borders.keys()}
        for lk in events_contained.keys():
            for event_i in events_contained[lk]:
                times_contained[lk].append((self.res_d['tBegin'][event_i], self.res_d['tEnd'][event_i]))
        return times_contained

    def get_num_events(self, compare_tuples, times_contained):
        num_events = copy.deepcopy(compare_tuples)
        num_events_norm = copy.deepcopy(compare_tuples)

        for axon_k in compare_tuples.keys():
            axon_event_start_l = [v[0] for v in times_contained[axon_k]]
            axon_event_start_np = np.array(axon_event_start_l, dtype=int)

            for axon_k_loc in compare_tuples[axon_k].keys():
                landmark_id = compare_tuples[axon_k][axon_k_loc]

                landmark_start_l = [v[0] for v in times_contained[landmark_id]]
                landmark_start_np = np.array(landmark_start_l, dtype=int)

                c = 0
                for landmark_start in landmark_start_l:
                    x = landmark_start - axon_event_start_np
                    if len(x[(x < self.delay_max) & (x >= self.delay_min)]):
                        c += 1

                #Get number of events for axon location
                num_events[axon_k][axon_k_loc] = c
                if len(landmark_start_l) == 0:
                    num_events_norm[axon_k][axon_k_loc] = c
                else:
                    num_events_norm[axon_k][axon_k_loc] = c / len(landmark_start_l)
        return num_events, num_events_norm

    def get_ddr_heatmaps(self, event_centroids, event_indices, delays_l):
        ddr_heatmap = np.zeros([self.input_shape[0], self.input_shape[1]])
        ddr_heatmap_count = np.zeros([self.input_shape[0], self.input_shape[1]])
        for event_centroid, event_i, delay in zip(event_centroids, event_indices, delays_l):
            event_locs = np.unravel_index((self.res_d['x2D'][event_i]), [self.input_shape[0], self.input_shape[1]], order='F')
            if delay is not None:
                ddr_heatmap[event_locs] += delay
                ddr_heatmap_count[event_locs] += 1
        ddr_heatmap = np.divide(ddr_heatmap, ddr_heatmap_count, where=ddr_heatmap_count !=0)
        #ddr_heatmap3[ddr_heatmap3 <= 0] = np.nan
        return ddr_heatmap, ddr_heatmap_count

    def get_values_heatmap_from_landmarks(self, ddr_heatmap, ddr_heatmap_count, landmark_locs):
        dc_landmark_ids = []
        dc_means = []
        dc_stds = []

        for landmark_id in landmark_locs.keys():
            valid_count_locs = np.where(ddr_heatmap_count != 0)
            count_valid = np.ravel_multi_index(valid_count_locs, [self.input_shape[0], self.input_shape[1]], order='F')
            landmark_valid = np.ravel_multi_index(landmark_locs[landmark_id],[self.input_shape[0], self.input_shape[1]], order='F')

            valid_locs = np.array(list(set(list(count_valid)) & set(list(landmark_valid))), dtype=np.int)
            valid_locs_tuples = np.unravel_index(valid_locs, [self.input_shape[0], self.input_shape[1]], order='F')

            vs = ddr_heatmap[valid_locs_tuples]
            m = np.mean(vs)
            s = np.std(vs)

            dc_landmark_ids.append(landmark_id)
            dc_means.append(m)
            dc_stds.append(s)

        dc_i = np.argsort(dc_means)
        dc_landmark_ids_s = np.array(dc_landmark_ids)[dc_i]
        dc_means_s = np.array(dc_means)[dc_i]
        dc_stds_s = np.array(dc_stds)[dc_i]
        return dc_landmark_ids_s, dc_means_s, dc_stds_s

    def get_event_axon_delays(self, event_times, axon_times, delay_min, delay_max):
        delays = []

        for event_time in event_times:
            diff_arr = event_time - axon_times
            diff_arr_interval = diff_arr[((diff_arr >= self.delay_min) & (diff_arr <= self.delay_max))]

            if len(diff_arr_interval) == 0:
                min_del = None
            else:
                min_del = np.min(diff_arr_interval)
            delays.append(min_del)
        return delays

    def setup_activity_ratios(self):
        self.activity_ratios = {}
        for k in self.event_subsets.keys():
            mask = np.ones(self.input_shape[0], self.input_shape[1])
            if 'border_mask' in self.res_d:
                mask = self.res_d['border_mask']

            self.activity_ratios[k] = np.sum(self.event_grids[k]) / (np.sum(mask) * len(self.indices_d[k]))

    def setup_grids(self):
        self.event_grids = {}
        self.event_grids_norm_rel = {}
        self.event_grids_dff = {}
        #Event grid per behavioural state
        for k in self.event_subsets.keys():
            self.event_grids[k] = aqua_utils.get_event_grid_from_x2D(self.res_d['x2D'][self.event_subsets[k]], (self.input_shape[0], self.input_shape[1]))
        #Event grid normalized (for per frame events) and applied to relative to 'default' behavioural state
        for k in self.event_subsets.keys():
            self.event_grids_norm_rel[k] = np.copy(self.event_grids[k]) / len(self.indices_d[k]) - (np.copy(self.event_grids['default']) / len(self.indices_d['default']))
        #Event grid dff
        for k in self.event_subsets.keys():
            self.event_grids_dff[k] = aqua_utils.get_dff_grid_from_x2D(self.res_d['x2D'][self.event_subsets[k]], self.res_d['dffMax2'][self.event_subsets[k]], ((self.input_shape[0], self.input_shape[1])))
