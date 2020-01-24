import os, sys, glob
import numpy as np
from preprocessing import analysis_pp
from analysis.general_utils import aqua_utils, saving_utils, duration_utils

class AstrocyteAnalyzer():
    def __init__(self, base_path, day, aqua_bound=True):
        self.base_path = base_path
        self.day = day
        self.aqua_bound = aqua_bound
        self.experiment_path = os.path.join(self.base_path, 'day_' + str(self.day))
        print('experiment path', self.experiment_path)
        self.max_delay = 50 #in frames
        self.duration_small = 11 #number of frames for small duration signals (6 frames is around 5.x sec)
        self.behaviour_base_path = 'roi_data_merged/other.csv'
        self.oscilloscope_base_path = 'oscilloscope/oscilloscope.csv'
        self.id = os.path.join(os.path.basename(os.path.normpath(self.base_path)), 'day_' + str(self.day))

        if aqua_bound:
            self.aqua_base_path = 'aqua_bound/aqua_merged/merged.pkl'
        else:
            self.aqua_base_path = 'aqua_no_bound/aqua_merged/merged.pkl'

        self.behaviours_path = os.path.join(self.experiment_path, self.behaviour_base_path)
        self.oscilloscope_path = os.path.join(self.experiment_path, self.oscilloscope_base_path)
        self.save_res_path = os.path.join(self.experiment_path, self.aqua_base_path)

        print('save res path', self.save_res_path)
        # Behavioural indices
        self.indices_d, self.roi_dict, l = analysis_pp.read_behaviour_indices(
                                                        self.behaviours_path,
                                                        self.oscilloscope_path)
        self.stick_bin, self.speed_bin, self.whisker_bin, self.pupil_values, self.speed_values = l

        ##############################################################################
        self.res_d, self.input_shape_l, self.fr_inv, self.spatial_res = saving_utils.load_pickle(self.save_res_path)
        self.res_d = aqua_utils.apply_res_d_filter(self.res_d, min_event_duration=2, min_um_size=2)
        self.fr = 1.0/self.fr_inv
        self.res_d['time'] = self.res_d['tEnd'] - self.res_d['tBegin']
        self.res_d['time_s'] = self.res_d['time'] / self.fr
        self.res_d['duration'] = self.res_d['tEnd'] - self.res_d['tBegin']
        self.border = self.res_d['border_mask']

        self.input_shape = self.input_shape_l[0]
        self.total_indices = len(self.indices_d['default'])
        self.total_events = len(self.res_d['tBegin'])

        #FIX FOR SPEED 7. #NOTE THIS IS NOT PROGRAMMATICALLY NICE
        #FIXING SPEED.PY SHOULD REPLACE THIS LINE WITH JUST self.speed_values * self.fr
        self.speed_values = self.speed_values * (0.5 / 360) * 4 * self.fr * (2 * np.pi * 7.5)**2

        self.pupil_values_old = self.pupil_values
        self.pupil_values_z = (self.pupil_values - np.mean(self.pupil_values)) / np.std(self.pupil_values)
        #Normalize pupil values in range [0, 1]
        self.pupil_values = (self.pupil_values - np.min(self.pupil_values))/ (np.max(self.pupil_values) - np.min(self.pupil_values))

        #How many frames in a single minute
        self.minute_frames = int(np.round(self.fr * 60))
        self.total_minutes = len(self.indices_d['default']) / self.minute_frames
        print(self.input_shape, self.fr_inv, self.spatial_res)
        print(self.res_d.keys())

        self.bright_spots = np.zeros([self.input_shape[0], self.input_shape[1]])
        self.ignore_events_l = []

        #Bright spots specific to dataset that should be mostly negatives
        if self.id == 'm181129_d190111_c001/day_0':
            for y in range(149, 172):
                for x in range(467, 483):
                    self.bright_spots[y, x] = 1

        if self.id == 'm190129_d190226_cx/day_0':
            for y in range(235,260):
                for x in range(240,270):
                    self.bright_spots[y, x] = 1

        if self.id == 'm190129_d190226_cx/day_2':
            for y in range(250,290):
                for x in range(250, 290):
                    self.bright_spots[y, x] = 1

        if self.id == 'm2000219_d190411_c003/day_1':
            for y in range(234,260):
                for x in range(480, 496):
                    self.bright_spots[y, x] = 1

        #Specific video segments to remove events from (e.g. df/f is unusually high)
        if self.id == 'm181129_d190222_c005/day_0':
            #df/f is just incorrect, choose to ignore the last video segment
            self.ignore_events_l.extend(np.where(self.res_d['event_i_video_segment'] == 8)[0])

        #Filter events that take place in bright spots
        self.setup_bright_spot_events()

        #Get event indices corresponding to each behavioural state
        print('Event subsets...')
        self.setup_event_subsets() #(self.event_subsets)

        #Get event grid per behavioural state
        #Event grid = number of events taking place per pixel, integrated over time
        print('Event grids...')
        self.setup_grids() #(self.event_grids, self.event_grids_norm_rel)
        print('Activity ratios...')
        self.setup_activity_ratios()
        #Setup signal durations
        print('Signal durations...')
        self.setup_signal_durations() #(self.all_durations_d)

        print('Finished init')
        self.print_id = '_'.join(self.id.split('/'))

    def setup_event_subsets(self):
        #Get event indices corresponding to each behavioural state
        self.event_subsets = aqua_utils.get_event_subsets(self.indices_d, self.res_d, after_i=0, before_i=0, to_print=False)
        self.event_subsets_old = np.copy(self.event_subsets)
        #For each event_subset we remove bright spot events bright_spot_events_l
        keys_to_del = []

        for k in self.event_subsets.keys():
            #print('IGNORE ENVETS??', self.ignore_events_l)
            self.event_subsets[k] = np.sort(list(set(self.event_subsets[k]) - set(self.bright_spot_events_l) - set(self.ignore_events_l)))

            if len(self.event_subsets[k]) == 0:
                keys_to_del.append(k)

        for k in keys_to_del:
            del self.event_subsets[k]
            #del self.indices_d[k]

    def setup_grids(self):
        self.event_grids = {}
        self.event_grids_norm_rel = {}
        self.event_grids_norm = {}
        self.event_grids_event_norm = {}
        self.event_grids_dff = {}

        self.event_grids_1min = {}
        self.event_grids_1min_dff = {}

        #These event grids are normalized appropriatelly for visual comparison of heatmaps
        self.event_grids_compare = {}
        self.event_grids_compare_dff = {}

        #Event grid per behavioural state (by default normalized to 1 minute of activity)
        for k in self.event_subsets.keys():
            print(len(self.event_subsets[k]), k)
            print('-----------------------------')
            self.event_grids[k] = aqua_utils.get_event_grid_from_x2D(self.res_d['x2D'][self.event_subsets[k]], (self.input_shape[0], self.input_shape[1]))

        for k in self.event_subsets.keys():
            self.event_grids_compare[k] = np.copy(self.event_grids[k])
            #1 normalize activity to 1 minute of frames
            self.event_grids_compare[k] = (self.event_grids_compare[k] / len(self.indices_d[k])) * self.minute_frames
            #2 normalize activity to per event activity
            self.event_grids_compare[k] = self.event_grids_compare[k] / len(self.event_subsets[k])
            #3 normalize activity to total area of all events
            #self.event_grids_compare[k] = self.event_grids_compare[k] / (np.sum([len(x2D_i) if isinstance(x2D_i, np.ndarray) else 1 for x2D_i in self.res_d['x2D']]))

            self.event_grids_1min[k] = np.copy(self.event_grids[k])
            self.event_grids_1min[k] = (self.event_grids_1min[k] / len(self.indices_d[k])) * self.minute_frames

        #Event grid normalized per frame (how many events per frame on each pixel)
        for k in self.event_subsets.keys():
            self.event_grids_norm[k] = np.copy(self.event_grids[k]) / len(self.indices_d[k])
        #Event grid normalized based on total number of events (grid sums to 1)
        for k in self.event_subsets.keys():
            self.event_grids_event_norm[k]= np.copy(self.event_grids[k]) / np.sum(self.event_grids[k])

        #Event grid normalized (for per frame events) and applied to relative to 'default' behavioural state
        for k in self.event_subsets.keys():
            self.event_grids_norm_rel[k] = np.copy(self.event_grids[k]) / len(self.indices_d[k]) - (np.copy(self.event_grids['default']) / len(self.indices_d['default']))
        #Event grid dff (by default normalized to 1 minute of activity) also normalized by total number of events
        for k in self.event_subsets.keys():
            self.event_grids_dff[k] = aqua_utils.get_dff_grid_from_x2D(self.res_d['x2D'][self.event_subsets[k]], self.res_d['dffMax2'][self.event_subsets[k]], ((self.input_shape[0], self.input_shape[1])))
        for k in self.event_subsets.keys():
            self.event_grids_compare_dff[k] = np.copy(self.event_grids_dff[k])
            self.event_grids_compare_dff[k] = (self.event_grids_compare_dff[k] / len(self.indices_d[k])) * self.minute_frames
            self.event_grids_compare_dff[k] = self.event_grids_compare_dff[k] / len(self.event_subsets[k])

            self.event_grids_1min_dff[k] = np.copy(self.event_grids_dff[k])
            self.event_grids_1min_dff[k] = (self.event_grids_1min_dff[k] / len(self.indices_d[k])) * self.minute_frames

            #self.event_grids_compare_dff[k] = self.event_grids_compare_dff[k] / (np.sum([len(x2D_i) if isinstance(x2D_i, np.ndarray) else 1 for x2D_i in self.res_d['x2D']]))

    def setup_signal_durations(self):
        #Signal durations
        self.all_durations_d = {}
        #Signal durations less than 6 frames
        self.event_subsets_hs = {}
        self.all_durations_d_hs = {}
        self.all_durations_d_hs_i = {}

        for k in self.event_subsets.keys():
            self.all_durations_d[k] = self.res_d['tEnd'][self.event_subsets[k]] - self.res_d['tBegin'][self.event_subsets[k]]

        for k in self.event_subsets.keys():
            self.all_durations_d_hs_i[k] = np.where(self.all_durations_d[k] <= self.duration_small)
            self.event_subsets_hs[k] = self.event_subsets[k][self.all_durations_d_hs_i[k]]
            self.all_durations_d_hs[k] = self.all_durations_d[k][self.all_durations_d_hs_i[k]]

        self.all_durations_class_d = duration_utils.signal_duration_split(self.all_durations_d, signal_duration_ranges=(1.0/3, 2.0/3, 1))

    def setup_activity_ratios(self):
        self.activity_ratios = {}
        for k in self.event_subsets.keys():
            mask = np.ones(self.input_shape[0], self.input_shape[1])
            if 'border_mask' in self.res_d:
                mask = self.res_d['border_mask']

            self.activity_ratios[k] = np.sum(self.event_grids[k]) / (np.sum(mask) * len(self.indices_d[k]))

    def setup_bright_spot_events(self):
        centroids = aqua_utils.get_event_centroids_from_x2D(self.res_d['x2D'], [self.input_shape[0], self.input_shape[1]])
        self.bright_spot_events_l = []
        for i, (y, x) in enumerate(centroids):
            if self.bright_spots[y, x] == 1:
                self.bright_spot_events_l.append(i)
