import h5py
import os, sys, glob
import numpy as np
import plotly.offline as offline
from preprocessing import analysis_pp
from analysis.general_utils import aqua_utils, saving_utils, plotly_utils, general_utils, compare_astro_utils, correlation_utils, stat_utils
from scipy.stats.stats import power_divergence
from scipy.stats import ttest_ind_from_stats
import csv
import scipy.signal as ss
import math
import time
from pandas import DataFrame
from scipy import optimize
import pandas as pd
import matplotlib.pyplot as plt

class AstrocytePlotter():
    def __init__(self, output_folder):
        self.output_folder = output_folder

        #For correlation plots
        self.filter_probs = [0.05, 0.10, 0.25]
        self.n_samples_corr_fake = 20
        self.num_frames_splits_l = [250, 500, 1000, 3000, 6000, 12000, 24000, 100000]
        self.num_frames_splits_m_l = [0.5, 1, 2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80]
        self.num_frames_splits_splits_m_l = [10, 15, 20, 25, 30, 35, 40]
        self.max_split_comparison_samples = 100

        self.behaviours_list_a = ['default', 'rest', 'running',
                                'running_start', 'running_before', 'stick',
                                'stick_start', 'stick_end', 'stick_expect',
                                'stick_rest', 'whisker_rest_stick', 'whisker_stick']
        self.behaviours_list_small = ['whisker_rest_stick', 'default', 'rest', 'running', 'stick']

    def setup_plot_folders(self, output_experiment_path):
        paths = ['borders', 'behaviour_heatmaps', 'behaviours_basic',
                'signal_delays', 'signal_durations', 'triplet', 'behaviour_activity',
                'behaviour_areas', 'signal_basic_samples', 'signal_behaviour_samples',
                'correlations', 'random_events', 'splits', 'splits_self', 'signal_amplitudes',
                'signal_proportion_delays', 'signal_stick_run_samples', 'splits_split_split',
                'triplet_bar', 'size_v_time_corr',
                'behaviour_heatmaps_threshold_with_random',
                'split_behaviour_grids']

        for p in paths:
            try:
                os.makedirs(os.path.join(output_experiment_path, 'plots' , p))
            except:
                pass

    def setup_file_folders(self, output_experiment_path):
        paths = ['correlations', 'csv']
        for p in paths:
            try:
                print(os.path.join(output_experiment_path, 'files', p))
                os.makedirs(os.path.join(output_experiment_path, 'files', p))
            except:
                print('Folder structure exists?')

    def setup_plot_folders_comparison(self, output_experiment_path_comparison):
        paths = ['behaviour_heatmaps', 'triplet', 'intersection', 'correlations', 'align',
                 'intersection_border_xcorr_aligned',]
        for p in paths:
            try:
                os.makedirs(os.path.join(output_experiment_path_comparison, 'plots', p))
            except:
                print('Folder structure exists?')

    def setup_file_folders_comparison(self, output_experiment_path_comparison):
        paths = ['correlations', 'csv']
        for p in paths:
            try:
                print(os.path.join(output_experiment_path_comparison, 'files', p))
                os.makedirs(os.path.join(output_experiment_path_comparison, 'files', p))
            except:
                print('Folder structure exists?')

    def setup_plot_folders_all_comparison(self, output_experiment_path_all_comparison):
        #print(output_experiment_path_all_comparison)
        paths = ['size_histogram_comparison', 'amplitude_histogram_comparison', 'duration_histogram_comparison',
                'size_histogram_bh_comparison', 'amplitude_histogram_bh_comparison', 'duration_histogram_bh_comparison',
                'activity_all', 'activity_all_number_minute', 'waterfall_together', 'signal_proportion_delays',
                'signal_proportion_delays_alt_average_proportions',
                'behaviour_heatmaps_V2_comparison_scale',
                'bar_rest_run_all',
                'bar_rest_rest_stick_all',
                'bar_run_run_stick_all',
                'dot_rest_run_pair_all',
                'bar_run_stick_run_transition_all',

                'rest_to_run_proportions_alt',
                'run_to_rest_proportions_alt',
                'run_stick_run_proportions_alt',

                'run_stick_run_proportions_alt_filter_max_3_frames',
                'run_stick_run_proportions_alt_filter_max_5_frames',

                'rest_to_run_amplitudes_default_alt',
                'rest_to_run_amplitudes_alt',
                'rest_to_run_durations_alt',
                'rest_to_run_sizes_alt',
                'rest_to_run_speed_alt',
                'rest_to_run_pupil_alt',

                'run_to_rest_amplitudes_default_alt',
                'run_to_rest_amplitudes_alt',
                'run_to_rest_durations_alt',
                'run_to_rest_sizes_alt',

                'rest_to_run_amplitudes_default_outlier_alt',
                'rest_to_run_amplitudes_outlier_alt',
                'rest_to_run_durations_outlier_alt',
                'rest_to_run_sizes_outlier_alt',

                'run_to_rest_amplitudes_default_outlier_alt',
                'run_to_rest_amplitudes_outlier_alt',
                'run_to_rest_durations_outlier_alt',
                'run_to_rest_sizes_outlier_alt',


                'run_to_rest_speed_alt',
                'run_to_rest_pupil_alt',

                'run_stick_run_amplitudes_default_alt',
                'run_stick_run_amplitudes_alt',
                'run_stick_run_durations_alt',
                'run_stick_run_sizes_alt',

                'run_stick_run_amplitudes_default_outlier_alt',
                'run_stick_run_amplitudes_outlier_alt',
                'run_stick_run_durations_outlier_alt',
                'run_stick_run_sizes_outlier_alt',

                'run_stick_run_speed_alt',
                'run_stick_run_pupil_alt',

                'run_stick_run_amplitudes_default_alt_filter_max_3_frames',
                'run_stick_run_amplitudes_alt_filter_max_3_frames',
                'run_stick_run_durations_alt_filter_max_3_frames',
                'run_stick_run_sizes_alt_filter_max_3_frames',
                'run_stick_run_speed_alt_filter_max_3_frames',
                'run_stick_run_pupil_alt_filter_max_3_frames',

                'run_stick_run_amplitudes_default_alt_filter_max_5_frames',
                'run_stick_run_amplitudes_alt_filter_max_5_frames',
                'run_stick_run_durations_alt_filter_max_5_frames',
                'run_stick_run_sizes_alt_filter_max_5_frames',
                'run_stick_run_speed_alt_filter_max_5_frames',
                'run_stick_run_pupil_alt_filter_max_5_frames',

                'all_amplitudes', 'all_durations', 'all_sizes',
                'all_amplitudes_filt_bh', 'all_durations_filt_bh', 'all_sizes_filt_bh',
                'correlations',
                'correlations_long_events',
                'correlations_short_events',
                'correlations_no_align',
                'correlations_no_align_long_events',
                'correlations_no_align_short_events',
                'correlations_csv',
                'correlations_long_events_csv',
                'correlations_short_events_csv',
                'correlations_no_align_csv',
                'correlations_no_align_long_events_csv',
                'correlations_no_align_short_events_csv',

                'control',
                'outliers',

                'triplet_dot_all',
                'size_v_time_corr_ALL',
                'speed_v_events_ALL',

                'split_correlation_all',
                'behaviour_over_recording',
                'pixel_distribution',
                'splits_self_all',
                ]
        data_paths = [
            'correlations',
            'correlations_long_events',
            'correlations_short_events',
            'correlations_no_align',
            'correlations_no_align_long_events',
            'correlations_no_align_short_events',
            'control',
            'outliers',
            'behaviour_ratios',
            'split_correlation_all',
            'splits_self_all'
        ]

        for p in paths:
            #print('Trying...', p)
            try:
                os.makedirs(os.path.join(output_experiment_path_all_comparison, 'plots', p))
            except:
                print('Folder structure exists?')
        for p in data_paths:
            try:
                os.makedirs(os.path.join(output_experiment_path_all_comparison, 'data', p))
            except:
                print('Folder structure exists?')

    def get_output_experiment_path(self, astroA):
        experiment_id = '/'.join(astroA.experiment_path.split('/')[-3:])
        output_experiment_path = os.path.join(self.output_folder, experiment_id)
        return output_experiment_path

    def plot_all_single(self, astroA):
        output_experiment_path = self.get_output_experiment_path(astroA)
        print('Making dirs', output_experiment_path)
        self.setup_plot_folders(output_experiment_path)
        '''
        print('Plotting behaviours basic...')
        #Behaviour basic
        figs_basic_plots = self.get_behaviour_basic_plots(astroA)
        for fig_k in figs_basic_plots.keys():
            saving_utils.save_plotly_fig(figs_basic_plots[fig_k], os.path.join(output_experiment_path, 'plots', 'behaviours_basic', '{}'.format(fig_k)), width=1000, height=400)


        print('Plotting random samples of signals...')
        fig_signals = self.get_signal_figs_samples(astroA, 20)

        for i, fig_signal in enumerate(fig_signals):
            fig_signal_path = os.path.join(output_experiment_path, 'plots', 'signal_basic_samples', 'signal_{}'.format(i))
            saving_utils.save_plotly_fig(fig_signal, fig_signal_path)


        print('Plotting random samples of signals on different behaviours...')
        fig_bk_signals = self.get_signal_bk_figs_samples(astroA, 3)
        for bk in fig_bk_signals.keys():
            for i, fig_bk_signal in enumerate(fig_bk_signals[bk]):
                fig_bk_signal_path = os.path.join(output_experiment_path, 'plots', 'signal_behaviour_samples', 'signal_{}-{}'.format(bk, i))
                saving_utils.save_plotly_fig(fig_bk_signal, fig_bk_signal_path)

        print('Plotting local signal samples with stick and running...')
        stick_run_sample_path = os.path.join(output_experiment_path, 'plots', 'signal_stick_run_samples')
        fig_stick_run_samples_l = self.get_stick_run_sample_figs(astroA)

        for i, sample_figs in enumerate(fig_stick_run_samples_l):
            saving_utils.save_plotly_fig(sample_figs[0], os.path.join(stick_run_sample_path, '{}-running'.format(i)))
            saving_utils.save_plotly_fig(sample_figs[1], os.path.join(stick_run_sample_path, '{}-stick'.format(i)))

            for j in range(min(10, len(sample_figs[2]))):
                saving_utils.save_plotly_fig(sample_figs[2][j], os.path.join(stick_run_sample_path, '{}-signal_{}'.format(i, j)))

        print('Plotting behaviour heatmaps...')
        #Behaviour heatmaps
        fig_heatmap_grids, fig_heatmap_dff_grids = self.get_behaviour_contour_plots(astroA)
        heatmap_grid_base_path = os.path.join(output_experiment_path, 'plots', 'behaviour_heatmaps')
        for k in fig_heatmap_grids.keys():
            saving_utils.save_plotly_fig(fig_heatmap_grids[k], os.path.join(heatmap_grid_base_path, k))
            saving_utils.save_plotly_fig(fig_heatmap_dff_grids[k], os.path.join(heatmap_grid_base_path, k + 'dff'))

        print('Plotting behaviour activity bar plot...')
        behaviour_activity_path = os.path.join(output_experiment_path, 'plots', 'behaviour_activity', 'activity')
        fig_behaviour_activity = self.get_behaviour_activity_plot(astroA)
        saving_utils.save_plotly_fig(fig_behaviour_activity, behaviour_activity_path, width=1200, height=800)


        print('Plotting behaviour event size bar plot...')
        behaviour_area_path = os.path.join(output_experiment_path, 'plots', 'behaviour_areas', 'areas')
        fig_behaviour_area = self.get_behaviour_area_plot(astroA)
        saving_utils.save_plotly_fig(fig_behaviour_area, behaviour_area_path)

        '''

        '''
        print('Plotting behaviour amplitude size bar plot...')
        behaviour_amplitude_path = os.path.join(output_experiment_path, 'plots', 'signal_amplitudes', 'amplitudes')
        fig_behaviour_amplitude = self.get_behaviour_amplitude_bar_plot(astroA)
        saving_utils.save_plotly_fig(fig_behaviour_amplitude, behaviour_amplitude_path)

        if astroA.aqua_bound == True:

            print('Plotting triplet plot...')
            #Triplet plot
            triplet_base_path = os.path.join(output_experiment_path, 'plots' , 'triplet')
            radii_path = os.path.join(output_experiment_path, 'plots', 'triplet', 'radii')
            fig_triplets, fig_radii_border = self.get_triplet_plots(astroA, n_bins=8)

            for k in fig_triplets.keys():
                saving_utils.save_plotly_fig(fig_triplets[k], os.path.join(triplet_base_path, k + '-triplet'))
            saving_utils.save_plotly_fig(fig_radii_border, radii_path)

            print('Plotting bar plots (triplet plot bands) num_events, duration, amplitude, ')
            measure_names = [None, 'Area', 'Amplitude', 'Time (s)']
            for bh in ['default', 'rest', 'running', 'stick', 'stick_rest', 'stick_run_ind_15']:
                for i, measure in enumerate([None, 'area', 'dffMax2', 'time_s']):
                    path = os.path.join(output_experiment_path, 'plots', 'triplet_bar', '{}_{}'.format(bh, measure))
                    if bh in astroA.event_subsets:
                        fig = self.triplet_bar_plot(astroA, bh=bh, measure=measure, n_bins=8, y_title=measure_names[i])
                        print('SAVING TRIPLET BAR')
                        saving_utils.save_plotly_fig(fig, path)
        '''
        '''
        print('Plotting signal durations...')
        #Signal durations plot
        durations_base_path = os.path.join(output_experiment_path, 'plots', 'signal_durations')
        fig_durations = self.get_signal_durations_plot(astroA)
        for k in fig_durations.keys():
            saving_utils.save_plotly_fig(fig_durations[k], os.path.join(durations_base_path, k + '-durations'))

        print('Plotting Signal duration split relative differences...')
        duration_split_differences_path = os.path.join(output_experiment_path, 'plots', 'signal_durations', 'duration_splits_relative_differences')
        fig_duration_split_differences = self.get_duration_split_differences_from_default(astroA)
        saving_utils.save_plotly_fig(fig_duration_split_differences, duration_split_differences_path)
        '''
        '''
        #Signal delays plot
        signal_delays_path = os.path.join(output_experiment_path, 'plots' , 'signal_delays')
        print('Plotting signal delays')
        fig_delays_waterfall_d, fig_delays_waterfall_interpolate_d = self.get_waterfall_delays_plot_all(astroA)

        for fig_k in fig_delays_waterfall_d.keys():
            print('FIG K', fig_k)
            saving_utils.save_plotly_fig(fig_delays_waterfall_d[fig_k], os.path.join(signal_delays_path, fig_k + '-delays_waterfall'))
            saving_utils.save_plotly_fig(fig_delays_waterfall_interpolate_d[fig_k], os.path.join(signal_delays_path, fig_k + '-delays_waterfall_interpolate'))
        '''
        '''
        print('Plotting borders')
        #Borders plot
        fig_border = self.get_border_plot(astroA)
        saving_utils.save_plotly_fig(fig_border, os.path.join(output_experiment_path, 'plots' , 'borders', 'border'))
        '''
        '''
        fig_proportion_delays_path = os.path.join(output_experiment_path, 'plots', 'signal_proportion_delays')
        fig_proportion_delays_d = self.get_proportion_delays_plot_all([astroA])

        for fig_k in fig_proportion_delays_d.keys():
            saving_utils.save_plotly_fig(fig_proportion_delays_d[fig_k], os.path.join(fig_proportion_delays_path, fig_k))
        '''
        '''
        print('Plotting sample frame split examples...')
        figs_frame_split_examples = self.get_frame_split_example_plots(astroA)
        for pk in figs_frame_split_examples.keys():
            for frame_split in figs_frame_split_examples[pk].keys():
                figs_frame_split_example_path = os.path.join(output_experiment_path, 'plots', 'correlations', 'frame_split_pair_example_frames_{}_p={}'.format(frame_split, pk))
                saving_utils.save_plotly_fig(figs_frame_split_examples[pk][frame_split], figs_frame_split_example_path)


        print('Plotting random astrocyte FULL sample plots...')
        figs_random_event_path = os.path.join(output_experiment_path, 'plots', 'random_events')
        fig_l = self.get_random_astrocyte_plot(astroA)
        for i, fig in enumerate(fig_l):
            saving_utils.save_plotly_fig(fig, os.path.join(figs_random_event_path, 'sample_{}'.format(i)))

        '''

        '''
        print('Plotting split counter')
        figs_frame_split = self.get_compare_frame_split_plots(astroA)

        for pk in figs_frame_split.keys():
            figs_frame_split_path = os.path.join(output_experiment_path, 'plots', 'splits', 'splits_p={}'.format(pk))
            saving_utils.save_plotly_fig(figs_frame_split[pk], figs_frame_split_path)



        #TODO RUN THIS
        print('Plotting frame split xcorr value to full self (self<->split)')
        fig_frame_split_self_path_a = os.path.join(output_experiment_path, 'plots', 'splits_self', 'splits_self_a')
        fig_frame_split_self_path_b = os.path.join(output_experiment_path, 'plots', 'splits_self', 'splits_self_b')
        fig_frame_split_self_a, fig_frame_split_self_b = self.get_compare_full_self_frame_split_plot_xcorr(astroA)
        saving_utils.save_plotly_fig(fig_frame_split_self_a, fig_frame_split_self_path_a)
        saving_utils.save_plotly_fig(fig_frame_split_self_b, fig_frame_split_self_path_b)
        '''

        '''
        print('Plotting frame split xcorr value to splits splits (split<->split)')
        fig_frame_split_self_path_a = os.path.join(output_experiment_path, 'plots', 'splits_split_split', 'splits_self_a')
        fig_frame_split_self_path_b = os.path.join(output_experiment_path, 'plots', 'splits_split_split', 'splits_self_b')
        fig_frame_split_self_a, fig_frame_split_self_b = self.get_compare_full_self_frame_split_split_plot_xcorr(astroA)
        saving_utils.save_plotly_fig(fig_frame_split_self_a, fig_frame_split_self_path_a)
        saving_utils.save_plotly_fig(fig_frame_split_self_b, fig_frame_split_self_path_b)
        '''
        '''
        print('Plotting first last 20 min of rest heatmap comparison...')
        fig_20min_rest_path = os.path.join(output_experiment_path, 'plots', 'splits_self', 'splits_first_last_rest_20min')
        fig_20min_rest = self.get_plot_first_last_x_min_behaviour(astroA, num_min=20, behaviour_ind='rest')
        if fig_20min_rest is not None:
            saving_utils.save_plotly_fig(fig_20min_rest, fig_20min_rest_path)

        print('Plotting continuous 20 min rest heatmaps compared to start...')
        fig_20min_cont_rest_path = os.path.join(output_experiment_path, 'plots', 'splits_self', 'cont_splits_first_last_rest_20min')
        fig_20min_cont_rest = self.get_plot_x_min_rest_relative(astroA, num_min=20, behaviour_ind='rest')
        if fig_20min_cont_rest is not None:
            saving_utils.save_plotly_fig(fig_20min_cont_rest, fig_20min_cont_rest_path)
        '''

        '''
        plt.ioff()
        print('Plotting Size vs Time correlation plot...')
        path = os.path.join(output_experiment_path, 'plots', 'size_v_time_corr')
        areas = np.log(astroA.res_d['area'])
        times = astroA.res_d['time_s']
        r, p = stat_utils.get_pearsonr(times, areas)

        df = pd.DataFrame({'Size': areas, 'Time': times})

        title ='Size vs Time correlation plot'
        text = 'r = {}, p < {}'.format(general_utils.truncate(r, 2), p)
        for kind in ['reg', 'hex', 'kde']:
            plotly_utils.seaborn_joint_grid(df, 'Size', 'Time', kind=kind, text=text)
            plt.savefig(os.path.join(path, '{}.svg'.format(kind)))
            plt.savefig(os.path.join(path, '{}.png'.format(kind)))
        '''

        print('Split BEHAVIOUR GRIDS...')
        n_chunks = 3
        for bh in ['default', 'running', 'rest']:
            event_grid_splits = aqua_utils.split_n_event_grids(astroA, bh=bh, n=n_chunks)
            path = os.path.join(output_experiment_path, 'plots', 'split_behaviour_grids')
            for i, event_grid_split in enumerate(event_grid_splits):
                plot = plotly_utils.plot_contour(event_grid_split, title='{}-split {}/{}'.format(bh, i+1, len(event_grid_splits)))
                saving_utils.save_plotly_fig(plot, os.path.join(path, 'bh_{}-split_{}-chunks_{}'.format(bh,i,n_chunks)))

        '''
        print('HEATMAPS V2_2... (each astro day scaled with random)')
        for dff_mode in ['False']:
            #for bh in ['default', 'running', 'rest', 'stick_run_ind_15', 'stick_rest']:
            for bh in ['default']:
                print('THIS REPETITION LOOP MUST BE ONCE')
                path = os.path.join(output_experiment_path, 'plots', 'behaviour_heatmaps_threshold_with_random')
                d = self.get_individual_heatmaps_threshold_scaled(astroA, bh=bh, threshold=0.7, num_samples=3, dff_mode=dff_mode)
                if d is None:
                    continue
                saving_utils.save_plotly_fig(d['contour'], os.path.join(path, 'bh_{}-dff_{}'.format(bh, dff_mode)))
                for i, contour_random in enumerate(d['contour_random']):
                    saving_utils.save_plotly_fig(contour_random, os.path.join(path, 'bh_{}-dff_{}-random_{}'.format(bh, dff_mode, i)))
        '''



#--------#--------#--------#--------#--------#--------#--------#--------#--------#--------
    #Experiment_id/days
    def plot_comparisons(self, astroA_l):
        output_experiment_path_comparison, days_str, day_l_s, astroA_l_s = self.setup_comparison_vars(astroA_l)
        print(output_experiment_path_comparison)
        #Setup folders
        self.setup_plot_folders_comparison(output_experiment_path_comparison)

        '''
        #Behaviour contour plots compare
        for k in astroA_l[0].event_subsets.keys():
            try:
                event_grids_l = [astroA.event_grids_compare[k] for astroA in astroA_l]
                fig_k = plotly_utils.plot_contour_multiple(event_grids_l, title=k + '_event grid comparison_' + days_str, height=500, width=600*len(astroA_l))
                saving_utils.save_plotly_fig(fig_k , os.path.join(output_experiment_path_comparison, 'plots', 'behaviour_heatmaps', k), height=500, width=600*len(astroA_l))
            except:
                continue
        for k in astroA_l[0].event_subsets.keys():
            try:
                event_grids_dff_l = [astroA.event_grids_compare_dff[k] for astroA in astroA_l]
                fig_k = plotly_utils.plot_contour_multiple(event_grids_dff_l, title=k + '_event grid dff comparison_' + days_str, height=500, width=600*len(astroA_l))
                saving_utils.save_plotly_fig(fig_k , os.path.join(output_experiment_path_comparison, 'plots', 'behaviour_heatmaps', k + '-dff'), height=500, width=600*len(astroA_l))
            except:
                continue
        '''

        '''
        name = '{}-{}'.format(astroA_l[0].day, astroA_l[1].day)

        behaviour_l = ['default', 'running', 'rest']
        p_l = [0.05, 0.1, 0.25]
        dff_mode_l = [False, True]

        for behaviour in behaviour_l:
            for dff_mode in dff_mode_l:
                for p in p_l:
                    same_spots_prob, astro_filt_l, astro_all_filt, astro_nz_bool_l, astro_all_nz_bool = compare_astro_utils.get_astro_pair_same_spots_prob([astroA_l[0], astroA_l[1]], p=0.05, dff_mode=True)

                    print('Plotting intersections...')
                    top_five_perc_path = os.path.join(output_experiment_path_comparison, 'plots', 'intersection', name + 'bh_{}-dff_{}-top_{}'.format(behaviour, dff_mode, p))
                    nz_border_path = os.path.join(output_experiment_path_comparison, 'plots', 'intersection', name + 'nz_border')
                    fig_perc = plotly_utils.plot_contour_multiple([astro_filt_l[0], astro_filt_l[1], astro_all_filt],
                                                                    subplot_titles=['top 5% values day {}'.format(astroA_l[0].day), 'top 5% values day {}'.format(astroA_l[1].day), 'intersection'],
                                                                    title='Probability to occur randomly {:.2e}'.format(same_spots_prob),
                                                                    color_bar_title='',
                                                                    line_width=0.1,
                                                                    font_size_individual=40,
                                                                    scale_equal=False)
                    fig_bord = plotly_utils.plot_contour_multiple([astro_nz_bool_l[0].astype(int), astro_nz_bool_l[1].astype(int), astro_all_nz_bool.astype(int)],
                                                                    subplot_titles=['non-0 values day {}'.format(astroA_l[0].day), 'non-0 values day {}'.format(astroA_l[1].day), 'intersection'],
                                                                    title='Event activity borders',
                                                                    color_bar_title='',
                                                                    line_width=0.1,
                                                                    font_size_individual=40,
                                                                    scale_equal=False)

                    saving_utils.save_plotly_fig(fig_perc, top_five_perc_path, width=2000, height=1000)
                    saving_utils.save_plotly_fig(fig_bord, nz_border_path, width=2000, height=1000)
        '''
        '''
        behaviour_l = ['default', 'running', 'rest']
        dff_mode_l = [False, True]
        p_l = [0.05, 0.10, 0.25]
        for behaviour in behaviour_l:
            print('Plotting intersections after alignment...')
            #move_vector = compare_astro_utils.get_move_vector_xcorr_default(astroA_l[0], astroA_l[1])
            move_vector = [0, 0]
            #p_l = [0.05, 0.1, 0.25]
            for dff_mode in dff_mode_l:
                for p in p_l:
                    same_spots_prob, astro_filt_l, astro_all_filt, astro_nz_bool_l, astro_all_nz_bool = compare_astro_utils.get_astro_pair_same_spots_prob([astroA_l[0], astroA_l[1]], p=0.05, move_vector=move_vector, dff_mode=True)

                    print('Plotting intersections...')
                    top_perc_path = os.path.join(output_experiment_path_comparison, 'plots', 'intersection_border_xcorr_aligned', name + 'bh_{}-dff_{}-top_{}'.format(behaviour, dff_mode, p))
                    fig_perc = plotly_utils.plot_contour_multiple([astro_filt_l[0], astro_filt_l[1], astro_all_filt],
                                                                    subplot_titles=['top 5% values day {}'.format(astroA_l[0].day), 'top 5% values day {}'.format(astroA_l[1].day), 'intersection'],
                                                                    title='Probability to occur randomly {:.2e}'.format(same_spots_prob),
                                                                    color_bar_title='',
                                                                    line_width=0.1,
                                                                    font_size_individual=40,
                                                                    scale_equal=False)
                    saving_utils.save_plotly_fig(fig_perc, top_perc_path, width=2000, height=1000)
        '''
        '''
        print('Plotting correlations compare...')
        figs_compare_corrs = self.get_compare_max_corrs_plots(astroA_l)

        for pk in figs_compare_corrs.keys():
            figs_compare_corrs_path = os.path.join(output_experiment_path_comparison, 'plots', 'correlations', 'max_correlations_compare_p={}'.format(pk))
            saving_utils.save_plotly_fig(figs_compare_corrs[pk], figs_compare_corrs_path)

        print('Plotting compare alignments intersection sizes...')
        figs_compare_align = self.get_compare_align_plots(astroA_l)
        for setting in figs_compare_align.keys():
            for pk in figs_compare_align[setting].keys():
                figs_compare_align_path = os.path.join(output_experiment_path_comparison, 'plots', 'align', 'align_compare_s={}_p={}'.format(setting, pk))
                saving_utils.save_plotly_fig(figs_compare_align[setting][pk], figs_compare_align_path)

        for behaviour in self.behaviours_list_small:
            if (behaviour in astroA_l[0].indices_d) and (behaviour in astroA_l[1].indices_d):
                print('Plotting compare alignments xcorr full... (Aligning borders then taking xcorr value of the 2 astrocytes. Then compare to random astrocyte plots)')
                figs_compare_align_xcorr = self.get_compare_align_plots_xcorr(astroA_l, align_setting='xcorr', dff_mode=False, behaviour=behaviour)
                figs_compare_align_xcorr_path = os.path.join(output_experiment_path_comparison, 'plots', 'align', 'align_compare_xcorr_values_full_{}'.format(behaviour))
                saving_utils.save_plotly_fig(figs_compare_align_xcorr, figs_compare_align_xcorr_path)

                print('Plotting compare alignments dff xcorr full... (Aligning borders then taking xcorr value of the 2 astrocytes. Then compare to random astrocyte plots)')
                figs_compare_align_xcorr_dff = self.get_compare_align_plots_xcorr(astroA_l, align_setting='xcorr', dff_mode=True, behaviour=behaviour)
                figs_compare_align_xcorr_dff_path = os.path.join(output_experiment_path_comparison, 'plots', 'align', 'align_compare_xcorr_values_full_dff_{}'.format(behaviour))
                saving_utils.save_plotly_fig(figs_compare_align_xcorr_dff, figs_compare_align_xcorr_dff_path)
            else:
                print('Behaviour {} not existent in astro'.format(behaviour))

        print('Plotting sample for comparison')
        #Make contour plot of astro1, astro2, sample_1, sample_2, sample_3
        figs_compare_samples = self.get_compare_corrs_samples_plots(astroA_l)
        for pk in figs_compare_samples.keys():
            for s in figs_compare_samples[pk].keys():
                path_s = os.path.join(output_experiment_path_comparison, 'plots', 'correlations', '{}_p={}'.format(s, pk))
                saving_utils.save_plotly_fig(figs_compare_samples[pk][s], path_s)

        behaviour_corr_path = os.path.join(output_experiment_path_comparison, 'plots', 'correlations', 'behaviour_corr')
        fig_behaviour_corr = self.get_plot_compare_behaviour_correlation(astroA_l)
        saving_utils.save_plotly_fig(fig_behaviour_corr, behaviour_corr_path)

        behaviour_corr_path = os.path.join(output_experiment_path_comparison, 'plots', 'correlations', 'behaviour_corr_dff')
        fig_behaviour_corr = self.get_plot_compare_behaviour_correlation(astroA_l, dff_mode=True)
        saving_utils.save_plotly_fig(fig_behaviour_corr, behaviour_corr_path)
        '''

    def plot_comparisons_all(self, astroA_l, astroA_l_pairs=None, astroA_l_good_pairs=None, astroA_l_good=None, astroA_long_l=None):
        output_experiment_path_all_comparison, _, _, astroA_l_s = self.setup_comparison_all_vars(astroA_l)
        print('Plotting sizes histogram dataset comparison for each behaviour')
        self.setup_plot_folders_all_comparison(output_experiment_path_all_comparison)

        bh_l = ['rest', 'stick_rest', 'running', 'stick_run_ind_15']

        astroA_l_filt = []
        bh_l_test = ['rest', 'running', 'stick_run_ind_15', 'stick_rest']
        for astroA in astroA_l:
            include = True
            for bh in bh_l_test:
                if bh not in astroA.indices_d.keys() or bh not in astroA.activity_ratios.keys():
                    include = False
                    print(':(', astroA.print_id, bh)
            if include:
                astroA_l_filt.append(astroA)

        day_0_1_pairs = []
        if astroA_l_pairs is not None:
            for astroA_l_pair in astroA_l_pairs:
                if astroA_l_pair[1].day == 1:
                    day_0_1_pairs.append(astroA_l_pair)

        '''
        print('Saving results of ratios running, rest, stick-running, stick-rest of each astrocyte in csv...')

        c = ['running', 'rest', 'stick_run_ind_15', 'stick_rest', 'total_time_s', 'total_time_m', 'avg_running_speed', 'avg_speed_global']
        c_n = ['running', 'rest', 'stick_run', 'stick_rest', 'total_time(s)', 'total_time(m)', 'avg_speed(cm/s)', 'avg_speed_global(cm/s)']
        astro_ratios_np = np.zeros([len(astroA_l), len(c)])
        r = [astroA.id for astroA in astroA_l]
        for i, astroA in enumerate(astroA_l):
            num_frames = len(astroA.indices_d['default'])
            num_seconds = num_frames / astroA.fr
            num_minutes = general_utils.truncate(num_seconds / 60.0, 2)
            num_seconds = general_utils.truncate(num_seconds, 2)
            for j, k in enumerate(c):
                if j == 4:
                    astro_ratios_np[i, j] = num_seconds
                    continue
                if j == 5:
                    astro_ratios_np[i, j] = num_minutes
                    continue
                if k not in astroA.indices_d:
                    if 'speed' in k:
                        if k == 'avg_running_speed':
                            astro_ratios_np[i, j] = np.mean(astroA.speed_values[astroA.speed_values!=0])
                        elif k == 'avg_speed_global':
                            astro_ratios_np[i, j] = np.mean(astroA.speed_values)
                    else:
                        print('Not exist', k, astroA.id)
                        astro_ratios_np[i, j] = 0
                        continue
                else:
                    astro_ratios_np[i, j] = general_utils.truncate(len(astroA.indices_d[k]) / num_frames, 3)

        behaviour_ratios_csv_path = os.path.join(output_experiment_path_all_comparison, 'data', 'behaviour_ratios', 'ratios.csv')
        DataFrame(astro_ratios_np, columns=c, index=r).to_csv(behaviour_ratios_csv_path)
        '''



        '''
        measure_l = ['time_s', 'dffMax2', 'area']
        measure_names = ['Duration(s)', 'Amplitude', 'Area']

        print('Calcium signal behaviour change over time')
        #How does calcium signals change over recording time?
        #1 sort events by time
        path = os.path.join(output_experiment_path_all_comparison, 'plots', 'behaviour_over_recording')
        for astroA in astroA_l:
            for i, measure in enumerate(measure_l):
                sorted_ev_i = np.argsort(astroA.res_d['tBegin'])
                x = []
                y = []

                for ev_i in sorted_ev_i:
                    x.append(ev_i)
                    y.append(astroA.res_d[measure][ev_i])

                fig = plotly_utils.plot_scatter(np.array(x), np.array(y) , mode='markers', title='scatter', x_title='', y_title='')
                plotly_utils.apply_fun_axis_fig(fig, lambda x : x / astroA.fr, axis='x')
                saving_utils.save_plotly_fig(fig, os.path.join(path, '{}-{}'.format(astroA.print_id, measure_names[i])))
        '''
        '''
        print('Speed over time...')
        path = os.path.join(output_experiment_path_all_comparison, 'plots', 'behaviour_over_recording')
        for astroA in astroA_l:
            fig = plotly_utils.plot_scatter(np.arange(len(astroA.speed_values)), astroA.speed_values, mode='lines')
            plotly_utils.apply_fun_axis_fig(fig, lambda x : x / astroA.fr, axis='x')
            saving_utils.save_plotly_fig(fig, os.path.join(path, '{}-speed'.format(astroA.print_id)))
        '''
        '''
        print('Individual behaviour distribution plots...')
        for n_bins in [10, 20, 40, 80]:
            #Size, amplitude, signal duration distribution plots over all datasets on different behaviours
            for bh in bh_l:
                plt_l = []
                pth_l = []
                for measure, min_measure, max_measure in [
                    ['area', None, 6],
                    ['area', None, None],
                    ['dffMax2', None, 5],
                    ['dffMax2', None, None],
                    ['duration', None, None],
                    ['duration', None, 50]
                ]:
                    try:
                        for with_max in [True, False]:
                            measure_name = aqua_utils.get_measure_names(measure)
                            fig_path = os.path.join(output_experiment_path_all_comparison, 'plots', '{}_histogram_comparison'.format(measure_name), '{}-nbins={}-min={}-max={}'.format(bh, n_bins, min_measure, max_measure))
                            plot, _, _ = self.measure_distribution_plot(astroA_l,  bh, measure=measure, num_bins=n_bins, max_measure=max_measure, min_measure=min_measure, measure_name=measure_name)

                            if measure == 'duration':
                                plotly_utils.apply_fun_axis_fig(plot, lambda x : x / astroA_l[0].fr, axis='x')

                            saving_utils.save_pth_plt_l_log([plot], [fig_path], axis='x')

                    except KeyError as e:
                        print('Got key error: some behaviour its fine {}'.format(e))
        '''

        '''
        print('Comparing behaviour distribution plots...')
        for n_bins in [10, 20, 40, 80]:
            print('NUM BINS:', n_bins)
            for behaviour_l in [bh_l, ['rest', 'running'], ['running', 'stick'], ['rest', 'stick_rest'], ['running', 'stick_run_ind_15']]:
                for measure, min_measure, max_measure in [
                    #['area', None, None],
                    #['area', None, 10],
                    #['area', None, 20],
                    #['area', None, 60],
                    #['area', None, 100],
                    #['area', 5, 60],
                    ['area', 3, 100],
                    #['dffMax2', None, None],
                    #['dffMax2', None, 2],
                    #['dffMax2', 0.6, 2],
                    #['dffMax2', None, 5],
                    #['dffMax2', 0.6, 5],
                    #['duration', None, None],
                    #['duration', None, 30],
                    #['duration', None, 100]
                ]:

                    for confidence in [True]:
                        measure_name = aqua_utils.get_measure_names(measure)
                        path = os.path.join(output_experiment_path_all_comparison, 'plots', '{}_histogram_bh_comparison'.format(measure_name), 'behaviours-{}-nbins={}-min={}-max={}-conf={}'.format('_'.join(behaviour_l), n_bins, min_measure, max_measure, confidence))
                        plot, stats_d = self.measure_distribution_bh_compare_plot(astroA_l, behaviour_l, measure=measure, num_bins=n_bins, min_measure=min_measure, max_measure=max_measure, measure_name=measure_name, confidence=confidence, with_stats=True)

                        if measure == 'duration':
                            plotly_utils.apply_fun_axis_fig(plot, lambda x : x / astroA_l[0].fr, axis='x')

                        saving_utils.save_pth_plt_l_log([plot], [path])

                        #Save results in text file
                        for i, name in enumerate(stats_d['names']):
                            #Create folder
                            data_folder_path = path
                            try:
                                os.makedirs(path)
                            except:
                                pass
                            temp_d = {k : stats_d[k][i] for k in stats_d.keys()}
                            saving_utils.save_csv_dict(temp_d, os.path.join(data_folder_path, '{}.csv'.format(name)), key_order=['names', 'x', 'mean', 'conf_95', 'std'])
                            np.savetxt(os.path.join(data_folder_path, '{}-data.csv'.format(name)), np.array(temp_d['data']).transpose(), delimiter=",")

                    for confidence in [True]:
                        measure_name = aqua_utils.get_measure_names(measure)
                        plot, stats_d = self.measure_distribution_bh_compare_plot_exponential_fit(astroA_l, behaviour_l, measure=measure, num_bins=n_bins, min_measure=min_measure, max_measure=max_measure, measure_name=measure_name, confidence=False, with_stats=True)
                        path = os.path.join(output_experiment_path_all_comparison, 'plots', '{}_histogram_bh_comparison'.format(measure_name), 'behaviours-{}-nbins={}-min={}-max={}-conf={}_EXPFIT'.format('_'.join(behaviour_l), n_bins, min_measure, max_measure, confidence))
                        if measure == 'duration':
                            plotly_utils.apply_fun_axis_fig(plot, lambda x : x / astroA_l[0].fr, axis='x')

                        #Save results in text file
                        for i, name in enumerate(stats_d['names']):
                            #Create folder
                            data_folder_path = path
                            try:
                                os.makedirs(path)
                            except:
                                pass
                            temp_d = {k : stats_d[k][i] for k in stats_d.keys()}

                            if len(name.split('__')) == 2:
                                tx_name = name.split('__')[0] + '_expfit'
                            else:
                                tx_name = name
                            print('TX NAME', name)
                            saving_utils.save_csv_dict(temp_d, os.path.join(data_folder_path, '{}.csv'.format(tx_name)), key_order=['names', 'x', 'mean', 'conf_95', 'std'])
                            np.savetxt(os.path.join(data_folder_path, '{}-data.csv'.format(tx_name)), np.array(temp_d['data']).transpose(), delimiter=",")
                        saving_utils.save_plotly_fig(plot, path)

                        print('THE STAT HERE?', stats_d)
        '''
        '''
        print('Violin plots...')

        plt_l = []
        pth_l = []

        for max_dff in [2, 5, 10, None]:
            #VIOLIN PLOTS comparing TWO behaviour distribution plots (but in violin form)
            fig_amp_violin_path = os.path.join(output_experiment_path_all_comparison, 'plots', 'amplitude_histogram_comparison', 'violin_rest_run_dff={}'.format(max_dff))
            fig = self.amplitude_distribution_plot_violin_duo(astroA_l_filt, 'rest', 'running', max_dff=max_dff)
            #saving_utils.save_plotly_fig(fig, fig_amp_violin_path)
            plt_l.append(fig)
            pth_l.append(fig_amp_violin_path)
            fig_amp_violin_path = os.path.join(output_experiment_path_all_comparison, 'plots', 'amplitude_histogram_comparison', 'violin_run_stick_dff={}'.format(max_dff))
            fig = self.amplitude_distribution_plot_violin_duo(astroA_l_filt, 'running', 'stick_run_ind_15', max_dff=max_dff)
            #saving_utils.save_plotly_fig(fig, fig_amp_violin_path2)
            plt_l.append(fig)
            pth_l.append(fig_amp_violin_path)
            fig_amp_violin_path3 = os.path.join(output_experiment_path_all_comparison, 'plots', 'amplitude_histogram_comparison', 'violin_rest_stick_dff={}'.format(max_dff))
            fig = self.amplitude_distribution_plot_violin_duo(astroA_l_filt, 'rest', 'stick_rest', max_dff=max_dff)
            #saving_utils.save_plotly_fig(fig, fig_amp_violin_path)
            plt_l.append(fig)
            pth_l.append(fig_amp_violin_path3)

        for max_area in [9, 20, 40, None]:
            sizes_violin_path = os.path.join(output_experiment_path_all_comparison, 'plots', 'sizes_histogram_comparison', 'violin_rest_run_area={}'.format(max_area))
            fig = self.sizes_distribution_plot_violin_duo(astroA_l_filt, 'rest', 'running', max_area=max_area)
            plt_l.append(fig)
            pth_l.append(sizes_violin_path)
            sizes_violin_path = os.path.join(output_experiment_path_all_comparison, 'plots', 'sizes_histogram_comparison', 'violin_run_stick_area={}'.format(max_area))
            fig = self.sizes_distribution_plot_violin_duo(astroA_l_filt, 'running', 'stick_run_ind_15', max_area=max_area)
            plt_l.append(fig)
            pth_l.append(sizes_violin_path)
            sizes_violin_path = os.path.join(output_experiment_path_all_comparison, 'plots', 'sizes_histogram_comparison', 'violin_rest_stick_area={}'.format(max_area))
            fig = self.sizes_distribution_plot_violin_duo(astroA_l_filt, 'rest', 'stick_rest', max_area=max_area)
            plt_l.append(fig)
            pth_l.append(sizes_violin_path)

        for max_duration in [10, 20, 30, 40, None]:
            duration_violin_path = os.path.join(output_experiment_path_all_comparison, 'plots', 'duration_histogram_comparison', 'violin_rest_run_duration={}'.format(max_duration))
            fig = self.signal_duration_distribution_plot_violin_duo(astroA_l_filt, 'rest', 'running', max_duration=max_duration)
            plt_l.append(fig)
            pth_l.append(duration_violin_path)

            duration_violin_path = os.path.join(output_experiment_path_all_comparison, 'plots', 'duration_histogram_comparison', 'violin_run_stick_duration={}'.format(max_duration))
            fig = self.signal_duration_distribution_plot_violin_duo(astroA_l_filt, 'running', 'stick_run_ind_15', max_duration=max_duration)
            plt_l.append(fig)
            pth_l.append(duration_violin_path)

            duration_violin_path = os.path.join(output_experiment_path_all_comparison, 'plots', 'duration_histogram_comparison', 'violin_rest_stick_duration={}'.format(max_duration))
            fig = self.signal_duration_distribution_plot_violin_duo(astroA_l_filt, 'rest', 'stick_rest', max_duration=max_duration)
            plt_l.append(fig)
            pth_l.append(duration_violin_path)

        save_pth_plt_l_log(plt_l, pth_l, axis='y')
        '''

        print('Splits SELF ALL')
        #STEP 1
        #Take only long duration astrocytes
        #Set maximum length of astrocyte duration to be 70min
        #Then apply splits with xcorr
        data_save_path = os.path.join(output_experiment_path_all_comparison, 'data', 'splits_self_all')
        path = os.path.join(output_experiment_path_all_comparison, 'plots', 'splits_self_all')
        y_l_l = []
        x_l = []
        minute_frame_splits_l = [35, 30, 25, 20, 15, 10, 5, 2]
        cut_duration = 70
        param_str = 'cut_{}-'.format(cut_duration) + 'splits_{}-'.format('_'.join([str(m) for m in minute_frame_splits_l]))

        name_l = []
        for i, astroA in enumerate(astroA_long_l):
            curr_save_path = os.path.join(data_save_path, 'id_{}-{}.pkl'.format(astroA.print_id, param_str))
            res_d = self.get_compare_full_self_results_alt(astroA, cut_duration_min=cut_duration, minute_frame_splits_l=minute_frame_splits_l, save_pkl_path=curr_save_path)
            y_l_l.append(res_d['y'])
            x_l.append(res_d['x'])
            name_l.append(astroA.print_id)

        fig, stats_d = plotly_utils.plot_scatter_mult_with_avg(x_l[0], y_l_l, None, name_l, mode='lines', title='Splits self', x_title='Splits (minutes)', y_title='Correlation',
                        xrange=None, yrange=None, confidence=True, with_stats=True, point_box=True)


        df_data_m = DataFrame(stats_d['mean_l_l'], columns=stats_d['x'], index=stats_d['names'])
        df_ci = DataFrame(stats_d['conf_95'], columns=stats_d['x'], index=stats_d['names'])
        df_mean = DataFrame([stats_d['mean'], stats_d['mean_conf']], columns=stats_d['x'], index=['mean', 'conf_95'])
        df_data_m.to_csv(path + '-data_means.csv')
        df_ci.to_csv(path + '-data_ci.csv')
        df_mean.to_csv(path + '-mean_and_CI.csv')

        saving_utils.save_plotly_fig(fig, path)
        '''
        print('HEATMAPS V2... (astro days scaled the same (to minimum maximum scale of the 2))')
        for astroA_pair in astroA_l_pairs:
            for dff_mode in ['False']:
                for bh in ['default', 'running', 'rest', 'stick_run_ind_15', 'stick_rest']:
                    path = os.path.join(output_experiment_path_all_comparison, 'plots', 'behaviour_heatmaps_V2_comparison_scale', self.get_astro_pair_id(astroA_pair))
                    d = self.get_day_heatmaps_scaled(astroA_pair, bh=bh, dff_mode=dff_mode)
                    if d is None:
                        continue
                    try:
                        os.makedirs(os.path.join(path))
                    except:
                        pass
                    saving_utils.save_plotly_fig(d['contour_0'], os.path.join(path, 'bh_{}-day_{}-dff_{}'.format(bh, astroA_pair[0].day, dff_mode)))
                    saving_utils.save_plotly_fig(d['contour_x'], os.path.join(path, 'bh_{}-day_{}-dff_{}'.format(bh, astroA_pair[1].day, dff_mode)))
        '''
        '''
        #TODO FIX THE DOT PLOTS
        #TODO CAN JUST ADD ANOTHER LOOP FOR THE BEHAVIOURS LOTS OF REPETITION
        bh_l_activity = ['rest', 'running', 'stick_rest', 'stick_run_ind_15']

        print('Bar charts and dot plots of all amplitudes, durations, sizes')

        #for type_plot in ['dot', 'bar']:
        for type_plot in ['bar']:
            for error_type in ['std', 'conf']:
                for err_symmetric in [True, False]:
                    path = os.path.join(output_experiment_path_all_comparison, 'plots', 'all_amplitudes', '{}_plot_dff_filter_event_{}_symm{}'.format(type_plot, error_type, err_symmetric))
                    fig, stats_d = self.get_all_signal_attribute_plot(astroA_l_s, bh_l_activity, type_plot=type_plot, type_event='dffMax2',
                                                                    y_title='Amplitude', title='Amplitudes', error_type=error_type, err_symmetric=err_symmetric, with_stats=True)
                    saving_utils.save_plotly_fig(fig, path)
                    saving_utils.save_csv_dict(stats_d, path + '.csv', key_order=['behaviour', 'mean', 'std', 'conf_95'])
                    saving_utils.save_csv_dict(stats_d['data'], path +'-data.csv', key_order=stats_d['behaviour'])
                    len_d = {k: [len(stats_d['data'][k])] for k in stats_d['data'].keys()}
                    saving_utils.save_csv_dict(len_d, path +'-len_data.csv', key_order=stats_d['behaviour'])

                    path = os.path.join(output_experiment_path_all_comparison, 'plots', 'all_amplitudes', '{}_plot_dff_notfiltered_{}_symm{}'.format(type_plot, error_type, err_symmetric))
                    fig, stats_d = self.get_all_signal_attribute_plot(astroA_l_s, bh_l_activity, type_plot=type_plot, type_event='dffMax',
                                                                    y_title='Amplitude', title='Amplitudes', error_type=error_type, err_symmetric=err_symmetric, with_stats=True)
                    saving_utils.save_plotly_fig(fig, path)
                    saving_utils.save_csv_dict(stats_d, path + '.csv', key_order=['behaviour', 'mean', 'std', 'conf_95'])
                    saving_utils.save_csv_dict(stats_d['data'], path +'-data.csv', key_order=stats_d['behaviour'])
                    len_d = {k: [len(stats_d['data'][k])] for k in stats_d['data'].keys()}
                    saving_utils.save_csv_dict(len_d, path +'-len_data.csv', key_order=stats_d['behaviour'])

                    path = os.path.join(output_experiment_path_all_comparison, 'plots', 'all_durations', '{}_plot_{}_symm{}'.format(type_plot, error_type, err_symmetric))
                    fig, stats_d = self.get_all_signal_attribute_plot(astroA_l_s, bh_l_activity, type_plot=type_plot, type_event='time_s',
                                                                        y_title='Duration (s)', title='Event durations', error_type=error_type, err_symmetric=err_symmetric, with_stats=True)
                    saving_utils.save_plotly_fig(fig, path)
                    saving_utils.save_csv_dict(stats_d, path + '.csv', key_order=['behaviour', 'mean', 'std', 'conf_95'])
                    saving_utils.save_csv_dict(stats_d['data'], path +'-data.csv', key_order=stats_d['behaviour'])
                    len_d = {k: [len(stats_d['data'][k])] for k in stats_d['data'].keys()}
                    saving_utils.save_csv_dict(len_d, path +'-len_data.csv', key_order=stats_d['behaviour'])

                    path = os.path.join(output_experiment_path_all_comparison, 'plots', 'all_sizes', '{}_plot_{}_symm{}'.format(type_plot, error_type, err_symmetric))
                    fig, stats_d = self.get_all_signal_attribute_plot(astroA_l_s, bh_l_activity, type_plot=type_plot, type_event='area',
                                                                    y_title='Event sizes (\u03bcm<sup>2</sup>)', title='Sizes of events', error_type=error_type, err_symmetric=err_symmetric, with_stats=True)
                    saving_utils.save_plotly_fig(fig, path)
                    saving_utils.save_csv_dict(stats_d, path + '.csv', key_order=['behaviour', 'mean', 'std', 'conf_95'])
                    saving_utils.save_csv_dict(stats_d['data'], path +'-data.csv', key_order=stats_d['behaviour'])
                    len_d = {k: [len(stats_d['data'][k])] for k in stats_d['data'].keys()}
                    saving_utils.save_csv_dict(len_d, path +'-len_data.csv', key_order=stats_d['behaviour'])
        '''
        '''
        print('COMPARE THIS', len(astroA_l_filt), 'WITH THIS', len(astroA_l_s))

        for astroA in astroA_l_filt:
            for bh_k in bh_l_activity:
                if bh_k not in astroA.event_subsets.keys():
                    print('SHOULD NOT HAPPEND BH ', bh_k, 'NOT IN', astroA.print_id)

        for type_plot in ['bar']:
            for error_type in ['std', 'conf']:
                for err_symmetric in [True, False]:
                    path = os.path.join(output_experiment_path_all_comparison, 'plots', 'all_amplitudes_filt_bh', '{}_plot_dff_filter_event_{}_symm{}'.format(type_plot, error_type, err_symmetric))
                    fig, stats_d = self.get_all_signal_attribute_plot(astroA_l_filt, bh_l_activity, type_plot=type_plot, type_event='dffMax2',
                                                                    y_title='Amplitude', title='Amplitudes', error_type=error_type, err_symmetric=err_symmetric, with_stats=True)
                    saving_utils.save_plotly_fig(fig, path)
                    saving_utils.save_csv_dict(stats_d, path + '.csv', key_order=['behaviour', 'mean', 'std', 'conf_95'])
                    saving_utils.save_csv_dict(stats_d['data'], path +'-data.csv', key_order=stats_d['behaviour'])

                    path = os.path.join(output_experiment_path_all_comparison, 'plots', 'all_amplitudes_filt_bh', '{}_plot_dff_notfiltered_{}_symm{}'.format(type_plot, error_type, err_symmetric))
                    fig, stats_d = self.get_all_signal_attribute_plot(astroA_l_filt, bh_l_activity, type_plot=type_plot, type_event='dffMax',
                                                                    y_title='Amplitude', title='Amplitudes', error_type=error_type, err_symmetric=err_symmetric, with_stats=True)
                    saving_utils.save_plotly_fig(fig, path)
                    saving_utils.save_csv_dict(stats_d, path + '.csv', key_order=['behaviour', 'mean', 'std', 'conf_95'])
                    saving_utils.save_csv_dict(stats_d['data'], path +'-data.csv', key_order=stats_d['behaviour'])

                    path = os.path.join(output_experiment_path_all_comparison, 'plots', 'all_durations_filt_bh', '{}_plot_{}_symm{}'.format(type_plot, error_type, err_symmetric))
                    fig, stats_d = self.get_all_signal_attribute_plot(astroA_l_filt, bh_l_activity, type_plot=type_plot, type_event='time_s',
                                                                        y_title='Duration (s)', title='Event durations', error_type=error_type, err_symmetric=err_symmetric, with_stats=True)
                    saving_utils.save_plotly_fig(fig, path)
                    saving_utils.save_csv_dict(stats_d, path + '.csv', key_order=['behaviour', 'mean', 'std', 'conf_95'])
                    saving_utils.save_csv_dict(stats_d['data'], path +'-data.csv', key_order=stats_d['behaviour'])

                    path = os.path.join(output_experiment_path_all_comparison, 'plots', 'all_sizes_filt_bh', '{}_plot_{}_symm{}'.format(type_plot, error_type, err_symmetric))
                    fig, stats_d = self.get_all_signal_attribute_plot(astroA_l_filt, bh_l_activity, type_plot=type_plot, type_event='area',
                                                                    y_title='Event sizes (\u03bcm<sup>2</sup>)', title='Sizes of events', error_type=error_type, err_symmetric=err_symmetric, with_stats=True)
                    saving_utils.save_plotly_fig(fig, path)
                    saving_utils.save_csv_dict(stats_d, path + '.csv', key_order=['behaviour', 'mean', 'std', 'conf_95'])
                    saving_utils.save_csv_dict(stats_d['data'], path +'-data.csv', key_order=stats_d['behaviour'])


        '''
        """
        print('--------------------------------------------------------------------------------------------------')
        print('Distribution of pixel values real vs fake...')
        path = os.path.join(output_experiment_path_all_comparison, 'plots', 'pixel_distribution')

        x_l = []
        y_l = []
        name_l = [astroA.print_id for astroA in astroA_l]
        for astroA in astroA_l:
            grid = astroA.event_grids_1min['default']
            grid = np.interp(grid, (grid.min(), grid.max()), (0, 1))
            grid_flat = grid.flatten()
            grid_flat_nz = grid_flat[grid_flat != 0]

            hist, bin_edges = np.histogram(grid_flat_nz, bins=20, range=(0,1), density=True)

            x_l = bin_edges[:-1]
            y_l.append(hist)

        y_l_fmt = []

        for i in range(len(y_l[0])):
            y_l_fmt.append([y[i] for y in y_l])

        plot_path = os.path.join(path, 'real')
        fig, stats_d = plotly_utils.plot_scatter_error(x_l, y_l_fmt, x_title='Pixel intensity percentile', y_title='Frequency (Density)', exp_fit=True, with_details=True)
        saving_utils.save_plotly_fig(fig, plot_path)

        df_data = DataFrame(np.array(stats_d['data']).T, columns=x_l, index=name_l)
        df_stats = DataFrame([stats_d['mean'], stats_d['conf_95'], stats_d['fit']], columns=x_l, index=['mean', 'conf_95', 'fit'])

        df_data.to_csv(plot_path + '-data.csv')
        df_stats.to_csv(plot_path +'-stats.csv')

        sample_l_all = []
        for astroA in astroA_l:
            d = self.get_individual_heatmaps_threshold_scaled(astroA, bh='default', threshold=1, num_samples=1, dff_mode=False, with_arr=True)
            sample_l_all.append(d['arrs_d']['arr_r'][0])

        x_l = []
        y_l = []

        for grid in sample_l_all:
            grid = np.interp(grid, (grid.min(), grid.max()), (0, 1))
            grid_flat = grid.flatten()
            grid_flat_nz = grid_flat[grid_flat != 0]
            #Normalize values to 1
            grid_flat_nz /= np.max(grid_flat_nz)
            hist, bin_edges = np.histogram(grid_flat_nz, bins=20, range=(0,1), density=True)
            x_l = bin_edges[:-1]
            y_l.append(hist)

        y_l_fmt = []

        for i in range(len(y_l[0])):
            y_l_fmt.append([y[i] for y in y_l])

        plot_path = os.path.join(path, 'fake')
        fig, stats_d = plotly_utils.plot_scatter_error(x_l, y_l_fmt, x_title='Pixel intensity percentile', y_title='Frequency (Density)', exp_fit=False, with_details=True)

        saving_utils.save_plotly_fig(fig, plot_path)

        df_data = DataFrame(np.array(stats_d['data']).T, columns=x_l)
        df_stats = DataFrame([stats_d['mean'], stats_d['conf_95']], columns=x_l, index=['mean', 'conf_95'])

        df_data.to_csv(plot_path + '-data.csv')
        df_stats.to_csv(plot_path +'-stats.csv')
        print('--------------------------------------------------------------------------------------------------')
        """
        '''
        print('SINGLE BAR CHART OF BEHAVIOURS (REST, RUN) of all astrocytes')
        names_l = ['amplitude', 'size', 'duration']
        measure_l = ['dffMax2', 'area', 'time_s' ]

        for i, measure in enumerate(measure_l):
            plot_path = os.path.join(output_experiment_path_all_comparison, 'plots', 'bar_rest_run_all', '{}'.format(names_l[i]))
            plot = self.get_measure_all_bar_plot(astroA_l, measure, bh_list=['rest', 'running'])
            saving_utils.save_plotly_fig(plot, plot_path)
        '''
        '''
        names_l = ['Event number (per minute)', 'amplitude', 'size', 'duration']
        measure_l = [None, 'dffMax2', 'area', 'time_s']
        bh_list_pairs = [['rest', 'running'], ['rest', 'stick_rest'], ['running', 'stick_run_ind_15']]
        bh_list_pairs_names = ['rest_run', 'rest_rest_stick', 'run_run_stick']

        for j, bh_list_pair in enumerate(bh_list_pairs):
            for i, measure in enumerate(measure_l):
                plot_path = os.path.join(output_experiment_path_all_comparison, 'plots', 'bar_{}_all'.format(bh_list_pairs_names[j]), '{}'.format('dots_'+names_l[i]))
                if 'stick_rest' in bh_list_pair:
                    plot, stats_d = self.get_measure_all_dot_plot(astroA_l_filt, measure, bh_list=bh_list_pair)
                else:
                    plot, stats_d = self.get_measure_all_dot_plot(astroA_l, measure, bh_list=bh_list_pair)
                saving_utils.save_plotly_fig(plot, plot_path)

                with open(os.path.join(plot_path + '.csv'), mode='w') as csv_file:
                    writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    l = ['']
                    l.extend(stats_d['x'])
                    l.extend(['conf_0', 'conf_1'])
                    writer.writerow(l)
                    for i in range(len(stats_d['names'])):
                        l = [stats_d['names'][i]]
                        l.extend(stats_d['mean_l_l'][i])
                        if 'conf_95' in stats_d:
                            l.extend(stats_d['conf_95'][i])
                        writer.writerow(l)

                    writer.writerow('')
                    writer.writerow(['mean_0', 'mean_1', 'mean_conf_0', 'mean_conf_1'])
                    l = []
                    l.extend(stats_d['mean'])
                    l.extend(stats_d['mean_conf'])
                    writer.writerow(l)
        '''
        """
        print('With transitions before and after measures dot plot')
        names_l = ['Event number (per minute)', 'amplitude', 'size', 'duration']
        measure_l = [None, 'dffMax2', 'area', 'time_s']
        delay_ranges_pairs = [ [3*astroA_l[0].fr, 6*astroA_l[0].fr],
                               #[1*astroA_l[0].fr, 1*astroA_l[0].fr],
                               #[2*astroA_l[0].fr, 4*astroA_l[0].fr]
                              ]
        delay_ranges_pairs = [[int(v[0]), int(v[1])] for v in delay_ranges_pairs]


        for delay_ranges_pair in delay_ranges_pairs:
            before_range, after_range = delay_ranges_pair
            for i, measure in enumerate(measure_l):
                plot_path = os.path.join(output_experiment_path_all_comparison, 'plots', 'bar_run_stick_run_transition_all', 'range_{}_{}_{}'.format(before_range, after_range, 'dots_'+names_l[i]))
                plot, stats_d = self.get_measure_all_transition_dot_plot(astroA_l, measure, before_bh='running_semi_exact',
                                                                            inds_bh='stick_exact_start', after_bh='running_semi_exact',
                                                                            before_range=before_range, after_range=after_range)
                saving_utils.save_plotly_fig(plot, plot_path)

                with open(os.path.join(plot_path + '.csv'), mode='w') as csv_file:
                    writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    l = ['']
                    l.extend(stats_d['x'])
                    l.extend(['conf_0', 'conf_1'])
                    writer.writerow(l)
                    for i in range(len(stats_d['names'])):
                        l = [stats_d['names'][i]]
                        l.extend(stats_d['mean_l_l'][i])
                        if 'conf_95' in stats_d:
                            l.extend(stats_d['conf_95'][i])
                        writer.writerow(l)

                    writer.writerow('')
                    writer.writerow(['mean_0', 'mean_1', 'mean_conf_0', 'mean_conf_1'])
                    l = []
                    l.extend(stats_d['mean'])
                    l.extend(stats_d['mean_conf'])
                    writer.writerow(l)
        """
        """
        #TODO ADD CSV
        bh_l_activity = ['rest', 'running', 'stick_rest', 'stick_run_ind_15']


        print('Activity all bar plot...')
        plot, stats_d = self.get_behaviour_activity_bar_plot_all(astroA_l_s, bh_l_activity, with_stats=True)
        plot_path = os.path.join(output_experiment_path_all_comparison, 'plots', 'activity_all', 'activity_bar')
        saving_utils.save_plotly_fig(plot, plot_path)

        print('Activity all number events per minute bar plot...')
        plot, stats_d = self.get_behaviour_activity_number_bar_plot_all(astroA_l_s, bh_l_activity, with_stats=True)
        plot_path = os.path.join(output_experiment_path_all_comparison, 'plots', 'activity_all_number_minute', 'activity_bar')
        saving_utils.save_plotly_fig(plot, plot_path)
        """

        '''
        bh_l_activity = ['rest', 'running', 'stick_rest', 'stick_run_ind_15']
        print('Activity all dot plot...')
        plot, stats_d = self.get_behaviour_activity_dot_plot_all(astroA_l_s, bh_l_activity)
        plot_path = os.path.join(output_experiment_path_all_comparison, 'plots', 'activity_all', 'activity_dot')
        saving_utils.save_plotly_fig(plot, plot_path)
        saving_utils.save_csv_dict(stats_d, plot_path+'.csv', key_order=['x', 'mean', 'conf_95'])
        print(stats_d['data'])
        #print(stats_d['data'].shape)
        DataFrame(stats_d['data'], columns=[astroA.print_id for astroA in astroA_l_s], index=stats_d['x']).to_csv(plot_path + '-data.csv')

        '''
        '''
        df_data_m = DataFrame(stats_d['mean_l_l'], columns=stats_d['x'], index=stats_d['names'])
        df_mean_conf = DataFrame([stats_d['mean'], stats_d['mean_conf']], columns=stats_d['x'], index=['mean', 'conf_95'])

        df_data_m.to_csv(path + '-data.csv')
        df_mean_conf.to_csv(path + '-mean_and_CI.csv')
        '''
        """
        print('Activity all dot plot with lines...')
        print(len(astroA_l_filt))

        plot, stats_d = self.get_behaviour_activity_dot_plot_all(astroA_l_filt, bh_l_activity, lines=True)
        plot_path = os.path.join(output_experiment_path_all_comparison, 'plots', 'activity_all', 'activity_dot_lines')
        saving_utils.save_plotly_fig(plot, plot_path)

        print('Activity all number events per minute dot plot...')
        plot, stats_d = self.get_behaviour_activity_number_dot_plot_all(astroA_l_s, bh_l_activity)
        plot_path = os.path.join(output_experiment_path_all_comparison, 'plots', 'activity_all_number_minute', 'activity_dot')
        saving_utils.save_plotly_fig(plot, plot_path)
        saving_utils.save_csv_dict(stats_d, plot_path+'.csv', key_order=['x', 'mean', 'conf_95'])

        print('Activity all number events per minute dot plot...')
        plot, stats_d = self.get_behaviour_activity_number_dot_plot_all(astroA_l_filt, bh_l_activity, lines=True)
        plot_path = os.path.join(output_experiment_path_all_comparison, 'plots', 'activity_all_number_minute', 'activity_dot_lines')
        saving_utils.save_plotly_fig(plot, plot_path)
        """
        '''
        print('Plotting bar plots (triplet plot bands) num_events, duration, amplitude for ALL TOGETHER')
        measure_names = [None, 'Area', 'Amplitude', 'Time (s)']
        for bh in ['default', 'rest', 'running', 'stick', 'stick_rest', 'stick_run_ind_15']:
            for i, measure in enumerate([None, 'area', 'dffMax2', 'time_s']):
                path = os.path.join(output_experiment_path_all_comparison, 'plots', 'triplet_dot_all', '{}_{}'.format(bh, measure))
                if bh in astroA.event_subsets:
                    fig, stats_d = self.triplet_dot_plot_all(astroA_l_s, bh=bh, measure=measure, n_bins=8, y_title=measure_names[i])
                    print('SAVING TRIPLET DOT ALL')
                    saving_utils.save_plotly_fig(fig, path)

                    print(stats_d.keys())
                    #Saving events only, we don't have CI's for each astrocyte
                    if measure is None:
                        df_data_m = DataFrame(stats_d['mean_l_l'], columns=stats_d['x'], index=stats_d['names'])
                        df_mean_conf = DataFrame([stats_d['mean'], stats_d['mean_conf']], columns=stats_d['x'], index=['mean', 'conf_95'])

                        df_data_m.to_csv(path + '-data.csv')
                        df_mean_conf.to_csv(path + '-mean_and_CI.csv')
                    else:
                        df_data_m = DataFrame(stats_d['mean_l_l'], columns=stats_d['x'], index=stats_d['names'])
                        df_ci = DataFrame(stats_d['conf_95'], columns=stats_d['x'], index=stats_d['names'])
                        df_mean = DataFrame([stats_d['mean'], stats_d['mean_conf']], columns=stats_d['x'], index=['mean', 'conf_95'])

                        df_data_m.to_csv(path + '-data_means.csv')
                        df_ci.to_csv(path + '-data_ci.csv')
                        df_mean.to_csv(path + '-mean_and_CI.csv')
        '''
        """
        #--------------------------------------------------
        #--------------------------------------------------
        #--------------------------------------------------
        ##REST TO RUN , RUN TO REST, RUN STICK RUN SECTION
        #--------------------------------------------------
        #--------------------------------------------------
        #--------------------------------------------------

        print('Alternative run-rest/rest-run averaging individual lines')
        delay_ranges_pairs = [ [3*astroA_l[0].fr, 6*astroA_l[0].fr],
                               [1*astroA_l[0].fr, 1*astroA_l[0].fr],
                               [2*astroA_l[0].fr, 4*astroA_l[0].fr]]
        delay_ranges_pairs = [[int(v[0]), int(v[1])] for v in delay_ranges_pairs]


        #measure_l = ['dffMax2default', 'dffMax2', 'time_s', 'area']
        #measure_path_l = ['amplitudes_default', 'amplitudes', 'durations', 'sizes']
        #measure_y_titles = ['Amplitude', 'Amplitude', 'Duration (s)', 'Size']

        measure_l = ['dffMax2default', 'time_s', 'area']
        measure_path_l = ['amplitudes_default', 'durations', 'sizes']
        measure_y_titles = ['Amplitude', 'Duration (s)', 'Size']

        measure_l = ['dffMax2default']
        measure_path_l = ['amplitudes_default']
        measure_y_titles = ['Amplitude']


        bh_measure_l = ['speed']
        bh_measure_path_l = ['speed']
        bh_measure_y_titles = ['Speed (cm/s)']

        print('Alt Proportion plots...')
        for delay_ranges_pair in delay_ranges_pairs:
            before_range, after_range = delay_ranges_pair
            for p in [#{'fit' : True, 'delay_step_size' : 1, 'confidence' : True},
                      #{'fit' : True, 'delay_step_size' : 5, 'confidence' : True},
                      {'fit' : True, 'delay_step_size' : 10, 'confidence': True}
                      ]:

                ################################################
                ##############Proportion plots##################
                ################################################

                print('EXTRA PARS', p, p.keys())

                print('rest to run prop')
                path = os.path.join(output_experiment_path_all_comparison, 'plots', 'rest_to_run_proportions_alt')
                fig_d, bin_stats = self.get_transition_proportion_delays_plot_all_alt(astroA_l, before_bh='rest_semi_exact', inds_bh='running_exact_start', after_bh='running_semi_exact',
                                                                                    before_range=before_range, after_range=after_range,
                                                                                    **p)
                for fig_k in fig_d.keys():
                    fig_id = os.path.join(path, fig_k + 'range_{}_{}-{}-fit_{}-step_{}-conf_{}'.format(before_range, after_range, fig_k, p['fit'], p['delay_step_size'], p['confidence']))
                    saving_utils.save_plotly_fig(fig_d[fig_k], fig_id)
                    saving_utils.save_csv_dict(bin_stats, path=fig_id + '.csv', key_order=['x', 'mean', 'std', 'confidence_95'])
                if p['delay_step_size'] == 10:
                    data_csv_path = os.path.join(path, 'range_{}_{}-step_{}-all.csv'.format(before_range, after_range, p['delay_step_size']))
                    DataFrame(bin_stats['y_all'], columns=bin_stats['x']).to_csv(data_csv_path, index=False)

                print('run to rest prop')
                path = os.path.join(output_experiment_path_all_comparison, 'plots', 'run_to_rest_proportions_alt')
                fig_d, bin_stats = self.get_transition_proportion_delays_plot_all_alt(astroA_l, before_bh='running_semi_exact', inds_bh='rest_start', after_bh='rest_semi_exact',
                                                                                    before_range=before_range, after_range=after_range,
                                                                                    **p)
                for fig_k in fig_d.keys():
                    fig_id = os.path.join(path, fig_k + 'range_{}_{}-{}-fit_{}-step_{}-conf_{}'.format(before_range, after_range, fig_k, p['fit'], p['delay_step_size'], p['confidence']))
                    saving_utils.save_plotly_fig(fig_d[fig_k], fig_id)
                    saving_utils.save_csv_dict(bin_stats, path=fig_id + '.csv', key_order=['x', 'mean', 'std', 'confidence_95'])

                if p['delay_step_size'] == 10:
                    data_csv_path = os.path.join(path, 'range_{}_{}-step_{}-all.csv'.format(before_range, after_range, p['delay_step_size']))
                    DataFrame(bin_stats['y_all'], columns=bin_stats['x']).to_csv(data_csv_path, index=False)

                print('run stick hit run prop')
                path = os.path.join(output_experiment_path_all_comparison, 'plots', 'run_stick_run_proportions_alt')
                fig_d, bin_stats = self.get_transition_proportion_delays_plot_all_alt(astroA_l, before_bh='running_semi_exact', inds_bh='stick_exact_start', after_bh='running_semi_exact',
                                                                                before_range=before_range, after_range=after_range,
                                                                                **p)
                for fig_k in fig_d:
                    fig_id = os.path.join(path, fig_k + 'range_{}_{}-{}-fit_{}-step_{}-conf_{}'.format(before_range, after_range, fig_k, p['fit'], p['delay_step_size'], p['confidence']))
                    saving_utils.save_plotly_fig(fig_d[fig_k], fig_id)
                    saving_utils.save_csv_dict(bin_stats, path=fig_id + '.csv', key_order=['x', 'mean', 'std', 'confidence_95'])


                if p['delay_step_size'] == 10:
                    data_csv_path = os.path.join(path, 'range_{}_{}-step_{}-all.csv'.format(before_range, after_range, p['delay_step_size']))
                    DataFrame(bin_stats['y_all'], columns=bin_stats['x']).to_csv(data_csv_path, index=False)


                print('run stick hit run prop duration filter [None, 3]')
                path = os.path.join(output_experiment_path_all_comparison, 'plots', 'run_stick_run_proportions_alt_filter_max_3_frames')
                fig_d, bin_stats = self.get_transition_proportion_delays_plot_all_alt(astroA_l, before_bh='running_semi_exact', inds_bh='stick_exact_start', after_bh='running_semi_exact',
                                                                                before_range=before_range, after_range=after_range, duration_filter=[None, 3],
                                                                                **p)
                for fig_k in fig_d:
                    fig_id = os.path.join(path, fig_k + 'range_{}_{}-{}-fit_{}-step_{}-conf_{}'.format(before_range, after_range, fig_k, p['fit'], p['delay_step_size'], p['confidence']))
                    saving_utils.save_plotly_fig(fig_d[fig_k], fig_id)
                    saving_utils.save_csv_dict(bin_stats, path=fig_id + '.csv', key_order=['x', 'mean', 'std', 'confidence_95'])

                if p['delay_step_size'] == 10:
                    data_csv_path = os.path.join(path, 'range_{}_{}-step_{}-all.csv'.format(before_range, after_range, p['delay_step_size']))
                    DataFrame(bin_stats['y_all'], columns=bin_stats['x']).to_csv(data_csv_path, index=False)

                print('run stick hit run prop duration filter [None, 5]')
                path = os.path.join(output_experiment_path_all_comparison, 'plots', 'run_stick_run_proportions_alt_filter_max_5_frames')
                fig_d, bin_stats = self.get_transition_proportion_delays_plot_all_alt(astroA_l, before_bh='running_semi_exact', inds_bh='stick_exact_start', after_bh='running_semi_exact',
                                                                                before_range=before_range, after_range=after_range, duration_filter=[None, 5],
                                                                                **p)
                for fig_k in fig_d:
                    fig_id = os.path.join(path, fig_k + 'range_{}_{}-{}-fit_{}-step_{}-conf_{}'.format(before_range, after_range, fig_k, p['fit'], p['delay_step_size'], p['confidence']))
                    saving_utils.save_plotly_fig(fig_d[fig_k], fig_id)
                    saving_utils.save_csv_dict(bin_stats, path=fig_id + '.csv', key_order=['x', 'mean', 'std', 'confidence_95'])

                if p['delay_step_size'] == 10:
                    data_csv_path = os.path.join(path, 'range_{}_{}-step_{}-all.csv'.format(before_range, after_range, p['delay_step_size']))
                    DataFrame(bin_stats['y_all'], columns=bin_stats['x']).to_csv(data_csv_path, index=False)

                '''
                ################################################
                ##############Measure plots#####################
                ################################################


                '''
                for m_i, measure in enumerate(measure_l):

                    print('rest to run measure: {}'.format(measure))
                    path = os.path.join(output_experiment_path_all_comparison, 'plots', 'rest_to_run_{}_alt'.format(measure_path_l[m_i]))

                    fig_d, bin_stats = self.get_transition_proportion_delays_plot_all_alt(astroA_l, before_bh='rest_semi_exact', inds_bh='running_exact_start', after_bh='running_semi_exact',
                                                                                        before_range=before_range, after_range=after_range,
                                                                                        measure=measure,
                                                                                        y_title=measure_y_titles[m_i],
                                                                                        **p)
                    for fig_k in fig_d.keys():
                        fig_id = os.path.join(path, fig_k + 'range_{}_{}-{}-fit_{}-step_{}-conf_{}'.format(before_range, after_range, fig_k, p['fit'], p['delay_step_size'], p['confidence']))
                        saving_utils.save_plotly_fig(fig_d[fig_k], fig_id)
                        saving_utils.save_csv_dict(bin_stats, path=fig_id + '.csv', key_order=['x', 'mean', 'std', 'confidence_95'])
                    if p['delay_step_size'] == 10:
                        data_csv_path = os.path.join(path, 'range_{}_{}-step_{}-all.csv'.format(before_range, after_range, p['delay_step_size']))
                        DataFrame(bin_stats['y_all'], columns=bin_stats['x']).to_csv(data_csv_path, index=False)

                    print('run to rest measure: {}'.format(measure))
                    path = os.path.join(output_experiment_path_all_comparison, 'plots', 'run_to_rest_{}_alt'.format(measure_path_l[m_i]))
                    fig_d, bin_stats = self.get_transition_proportion_delays_plot_all_alt(astroA_l, before_bh='running_semi_exact', inds_bh='rest_start', after_bh='rest_semi_exact',
                                                                                    before_range=before_range, after_range=after_range,
                                                                                    measure=measure,
                                                                                    y_title=measure_y_titles[m_i],
                                                                                    **p)

                    for fig_k in fig_d.keys():
                        fig_id = os.path.join(path, fig_k + 'range_{}_{}-{}-fit_{}-step_{}-conf_{}'.format(before_range, after_range, fig_k, p['fit'], p['delay_step_size'], p['confidence']))
                        saving_utils.save_plotly_fig(fig_d[fig_k], fig_id)
                        saving_utils.save_csv_dict(bin_stats, path=fig_id + '.csv', key_order=['x', 'mean', 'std', 'confidence_95'])
                    if p['delay_step_size'] == 10:
                        data_csv_path = os.path.join(path, 'range_{}_{}-step_{}-all.csv'.format(before_range, after_range, p['delay_step_size']))
                        DataFrame(bin_stats['y_all'], columns=bin_stats['x']).to_csv(data_csv_path, index=False)


                    print('run stick hit run measure: {}'.format(measure))
                    path = os.path.join(output_experiment_path_all_comparison, 'plots', 'run_stick_run_{}_alt'.format(measure_path_l[m_i]))
                    fig_d, bin_stats = self.get_transition_proportion_delays_plot_all_alt(astroA_l, before_bh='running_semi_exact', inds_bh='stick_exact_start', after_bh='running_semi_exact',
                                                                                    before_range=before_range, after_range=after_range,
                                                                                    measure=measure,
                                                                                    y_title=measure_y_titles[m_i],
                                                                                    **p)
                    for fig_k in fig_d.keys():
                        fig_id = os.path.join(path, fig_k + 'range_{}_{}-{}-fit_{}-step_{}-conf_{}'.format(before_range, after_range, fig_k, p['fit'], p['delay_step_size'], p['confidence']))
                        saving_utils.save_plotly_fig(fig_d[fig_k], fig_id)
                        saving_utils.save_csv_dict(bin_stats, path=fig_id + '.csv', key_order=['x', 'mean', 'std', 'confidence_95'])
                    if p['delay_step_size'] == 10:
                        data_csv_path = os.path.join(path, 'range_{}_{}-step_{}-all.csv'.format(before_range, after_range, p['delay_step_size']))
                        DataFrame(bin_stats['y_all'], columns=bin_stats['x']).to_csv(data_csv_path, index=False)

                    print('run stick hit run measure: max frames 3 {}'.format(measure))
                    path = os.path.join(output_experiment_path_all_comparison, 'plots', 'run_stick_run_{}_alt_filter_max_3_frames'.format(measure_path_l[m_i]))
                    fig_d, bin_stats = self.get_transition_proportion_delays_plot_all_alt(astroA_l, before_bh='running_semi_exact', inds_bh='stick_exact_start', after_bh='running_semi_exact',
                                                                                    before_range=before_range, after_range=after_range,
                                                                                    measure=measure,
                                                                                    y_title=measure_y_titles[m_i], duration_filter=[None, 3],
                                                                                    **p)
                    for fig_k in fig_d.keys():
                        fig_id = os.path.join(path, fig_k + 'range_{}_{}-{}-fit_{}-step_{}-conf_{}'.format(before_range, after_range, fig_k, p['fit'], p['delay_step_size'], p['confidence']))
                        saving_utils.save_plotly_fig(fig_d[fig_k], fig_id)
                        saving_utils.save_csv_dict(bin_stats, path=fig_id + '.csv', key_order=['x', 'mean', 'std', 'confidence_95'])
                    if p['delay_step_size'] == 10:
                        data_csv_path = os.path.join(path, 'range_{}_{}-step_{}-all.csv'.format(before_range, after_range, p['delay_step_size']))
                        DataFrame(bin_stats['y_all'], columns=bin_stats['x']).to_csv(data_csv_path, index=False)

                    print('run stick hit run measure: max frames 5 {}'.format(measure))
                    path = os.path.join(output_experiment_path_all_comparison, 'plots', 'run_stick_run_{}_alt_filter_max_5_frames'.format(measure_path_l[m_i]))
                    fig_d, bin_stats = self.get_transition_proportion_delays_plot_all_alt(astroA_l, before_bh='running_semi_exact', inds_bh='stick_exact_start', after_bh='running_semi_exact',
                                                                                    before_range=before_range, after_range=after_range,
                                                                                    measure=measure,
                                                                                    y_title=measure_y_titles[m_i], duration_filter=[None, 5],
                                                                                    **p)
                    for fig_k in fig_d.keys():
                        fig_id = os.path.join(path, fig_k + 'range_{}_{}-{}-fit_{}-step_{}-conf_{}'.format(before_range, after_range, fig_k, p['fit'], p['delay_step_size'], p['confidence']))
                        saving_utils.save_plotly_fig(fig_d[fig_k], fig_id)
                        saving_utils.save_csv_dict(bin_stats, path=fig_id + '.csv', key_order=['x', 'mean', 'std', 'confidence_95'])
                    if p['delay_step_size'] == 10:
                        data_csv_path = os.path.join(path, 'range_{}_{}-step_{}-all.csv'.format(before_range, after_range, p['delay_step_size']))
                        DataFrame(bin_stats['y_all'], columns=bin_stats['x']).to_csv(data_csv_path, index=False)

                ################################################
                ##############Behaviour measure plots###########
                ################################################

                for m_i, bh_measure in enumerate(bh_measure_l):

                    print('BH measure {} rest-run'.format(bh_measure))
                    path = os.path.join(output_experiment_path_all_comparison, 'plots', 'rest_to_run_{}_alt'.format(bh_measure_path_l[m_i]))
                    fig_d, bin_stats = self.get_transition_bh_values_plot_all_alt(astroA_l,
                                                                before_bh='rest_semi_exact', inds_bh='running_exact_start', after_bh='running_semi_exact',
                                                                bh_measure=bh_measure,
                                                                before_range=before_range, after_range=after_range,
                                                                y_title=bh_measure_y_titles[m_i],
                                                                **p)
                    for fig_k in fig_d.keys():
                        fig_id = os.path.join(path, fig_k + 'range_{}_{}-{}-fit_{}-step_{}-conf_{}'.format(before_range, after_range, fig_k, p['fit'], p['delay_step_size'], p['confidence']))
                        saving_utils.save_plotly_fig(fig_d[fig_k], fig_id)
                        saving_utils.save_csv_dict(bin_stats, path=fig_id + '.csv', key_order=['x', 'mean', 'std', 'confidence_95'])
                    if p['delay_step_size'] == 10:
                        data_csv_path = os.path.join(path, 'range_{}_{}-step_{}-all.csv'.format(before_range, after_range, p['delay_step_size']))
                        DataFrame(bin_stats['y_all'], columns=bin_stats['x']).to_csv(data_csv_path, index=False)

                    print('BH measure {} run-rest'.format(bh_measure))

                    path = os.path.join(output_experiment_path_all_comparison, 'plots', 'run_to_rest_{}_alt'.format(bh_measure_path_l[m_i]))
                    fig_d, bin_stats = self.get_transition_bh_values_plot_all_alt(astroA_l,
                                                                before_bh='running_semi_exact', inds_bh='rest_start', after_bh='rest_semi_exact',
                                                                bh_measure=bh_measure,
                                                                before_range=before_range, after_range=after_range,
                                                                y_title=bh_measure_y_titles[m_i],
                                                                **p)

                    for fig_k in fig_d.keys():
                        fig_id = os.path.join(path, fig_k + 'range_{}_{}-{}-fit_{}-step_{}-conf_{}'.format(before_range, after_range, fig_k, p['fit'], p['delay_step_size'], p['confidence']))
                        saving_utils.save_plotly_fig(fig_d[fig_k], fig_id)
                        saving_utils.save_csv_dict(bin_stats, path=fig_id + '.csv', key_order=['x', 'mean', 'std', 'confidence_95'])
                    if p['delay_step_size'] == 10:
                        data_csv_path = os.path.join(path, 'range_{}_{}-step_{}-all.csv'.format(before_range, after_range, p['delay_step_size']))
                        DataFrame(bin_stats['y_all'], columns=bin_stats['x']).to_csv(data_csv_path, index=False)
                    print('BH measure {} run-stick-run'.format(bh_measure))

                    path = os.path.join(output_experiment_path_all_comparison, 'plots', 'run_stick_run_{}_alt'.format(bh_measure_path_l[m_i]))
                    fig_d, bin_stats = self.get_transition_bh_values_plot_all_alt(astroA_l,
                                                                before_bh='running_semi_exact', inds_bh='stick_exact_start', after_bh='running_semi_exact',
                                                                bh_measure=bh_measure,
                                                                before_range=before_range, after_range=after_range,
                                                                y_title=bh_measure_y_titles[m_i],
                                                                **p)
                    for fig_k in fig_d.keys():
                        fig_id = os.path.join(path, fig_k + 'range_{}_{}-{}-fit_{}-step_{}-conf_{}'.format(before_range, after_range, fig_k, p['fit'], p['delay_step_size'], p['confidence']))
                        saving_utils.save_plotly_fig(fig_d[fig_k], fig_id)
                        saving_utils.save_csv_dict(bin_stats, path=fig_id + '.csv', key_order=['x', 'mean', 'std', 'confidence_95'])
                    if p['delay_step_size'] == 10:
                        data_csv_path = os.path.join(path, 'range_{}_{}-step_{}-all.csv'.format(before_range, after_range, p['delay_step_size']))
                        DataFrame(bin_stats['y_all'], columns=bin_stats['x']).to_csv(data_csv_path, index=False)
        """

        """
        print('OUTLIERS TRANSITION PLOTS...')
        delay_ranges_pairs = [ [3*astroA_l[0].fr, 6*astroA_l[0].fr],
                               [1*astroA_l[0].fr, 1*astroA_l[0].fr],
                               [2*astroA_l[0].fr, 4*astroA_l[0].fr]
                              ]

        delay_ranges_pairs = [[int(v[0]), int(v[1])] for v in delay_ranges_pairs]
        measure_l = ['dffMax2default', 'time_s', 'area']
        measure_path_l = ['amplitudes_default', 'durations', 'sizes']
        measure_y_titles = ['Amplitude', 'Duration (s)', 'Size']

        for delay_ranges_pair in delay_ranges_pairs:
            before_range, after_range = delay_ranges_pair
            for m_i, measure in enumerate(measure_l):
                print('rest to run measure: {}'.format(measure))
                path = os.path.join(output_experiment_path_all_comparison, 'plots', 'rest_to_run_{}_outlier_alt'.format(measure_path_l[m_i]))

                fig, stats_d = self.get_transition_outliers_plot(astroA_l, before_bh='rest_semi_exact', inds_bh='running_exact_start', after_bh='running_semi_exact',
                                                                                    before_range=before_range, after_range=after_range,
                                                                                    measure=measure,
                                                                                    y_title=measure_y_titles[m_i])

                fig_id = os.path.join(path, 'outlier_range_{}_{}'.format(before_range, after_range))
                saving_utils.save_plotly_fig(fig, fig_id)
                with open(os.path.join(fig_id + '.csv'), mode='w') as csv_file:
                    writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    for i in range(len(stats_d['names'])):
                        l = [stats_d['names'][i]]
                        l.extend(stats_d['mean'][i])
                        writer.writerow(l)


                print('run to rest measure: {}'.format(measure))
                path = os.path.join(output_experiment_path_all_comparison, 'plots', 'run_to_rest_{}_outlier_alt'.format(measure_path_l[m_i]))
                fig, stats_d = self.get_transition_outliers_plot(astroA_l, before_bh='running_semi_exact', inds_bh='rest_start', after_bh='rest_semi_exact',
                                                                                before_range=before_range, after_range=after_range,
                                                                                measure=measure,
                                                                                y_title=measure_y_titles[m_i])
                fig_id = os.path.join(path, 'outlier_range_{}_{}'.format(before_range, after_range))
                saving_utils.save_plotly_fig(fig, fig_id)
                with open(os.path.join(fig_id + '.csv'), mode='w') as csv_file:
                    writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    for i in range(len(stats_d['names'])):
                        l = [stats_d['names'][i]]
                        l.extend(stats_d['mean'][i])
                        writer.writerow(l)


                print('run stick hit run measure: {}'.format(measure))
                path = os.path.join(output_experiment_path_all_comparison, 'plots', 'run_stick_run_{}_outlier_alt'.format(measure_path_l[m_i]))
                fig, stats_d = self.get_transition_outliers_plot(astroA_l, before_bh='running_semi_exact', inds_bh='stick_exact_start', after_bh='running_semi_exact',
                                                                                before_range=before_range, after_range=after_range,
                                                                                measure=measure,
                                                                                y_title=measure_y_titles[m_i])

                fig_id = os.path.join(path, 'outlier_range_{}_{}'.format(before_range, after_range))
                saving_utils.save_plotly_fig(fig, fig_id)
                with open(os.path.join(fig_id + '.csv'), mode='w') as csv_file:
                    writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    for i in range(len(stats_d['names'])):
                        l = [stats_d['names'][i]]
                        l.extend(stats_d['mean'][i])
                        writer.writerow(l)
        """

        """
        print('Correlation plots ALL')
        if astroA_l_pairs is not None:
            for dff_mode in [False]:
                #for align_setting in ['xcorr', 'xcorr_free']:
                for align_setting in ['xcorr']:
                    #for filter_duration in [[None, None], [None, 1], [1, None]]:
                    for filter_duration in [[None, None], [None, 1], [1, None]]:
                        for bh in ['default', 'rest', 'running', 'stick']:
                            main_folder_id = 'correlations_no_align' if align_setting == 'xcorr_free' else 'correlations'
                            if (filter_duration[0] == None and filter_duration[1] == 1):
                                main_folder_id += '_short_events'
                            if (filter_duration[0] == 1 and filter_duration[1] == None):
                                main_folder_id += '_long_events'
                            fig_corr_path = os.path.join(output_experiment_path_all_comparison, 'plots', main_folder_id, 'xcorr_compare_{}_is_dff_{}'.format(bh, dff_mode))
                            save_results_path = os.path.join(output_experiment_path_all_comparison, 'data', main_folder_id, 'xcorr_compare_{}_is_dff_{}'.format(bh, dff_mode))
                            fig, pair_fakes_before, pair_fakes, pair_corrs_l_before, pair_corrs_l, days_id_l = self.get_compare_align_plot_xcorr_all(astroA_l_pairs, align_setting='xcorr', dff_mode=dff_mode, behaviour=bh, n_fake_samples=25 ,save_results_path=save_results_path)
                            saving_utils.save_plotly_fig(fig, fig_corr_path)

                            csv_corr_path = os.path.join(output_experiment_path_all_comparison, 'plots', main_folder_id + '_csv', 'xcorr_compare_{}_is_dff_{}.csv'.format(bh, dff_mode))
                            self.save_xcorr_pairs_align_results_csv(csv_corr_path, astroA_l_pairs, pair_fakes_before, pair_fakes, pair_corrs_l_before, pair_corrs_l)
        """
        '''

        print('Correlation plots (rest 0-1, run 0-1, rest-stick 0-1, run-stick 0-1, all, random)')
        file_id = 'xcorr_compare_states_all'
        if astroA_l_pairs is not None:
            for dff_mode in [False]:
                #for align_setting in ['xcorr', 'xcorr_free']:
                for align_setting in ['xcorr']:
                    for astro_pair in astroA_l_pairs:
                        #for filter_duration in [[None, None], [None, 1], [1, None]]:
                        for filter_duration in [[None, None]]:
                            main_folder_id = 'correlations_no_align' if align_setting == 'xcorr_free' else 'correlations'
                            if (filter_duration[0] == None and filter_duration[1] == 1):
                                main_folder_id += '_short_events'
                            if (filter_duration[0] == 1 and filter_duration[1] == None):
                                main_folder_id += '_long_events'
                            fig_corr_path = os.path.join(output_experiment_path_all_comparison, 'plots', main_folder_id, 'pair_{}_type_{}_is_dff_{}'.format(self.get_astro_pair_id(astro_pair), file_id, dff_mode))
                            save_pkl_path = os.path.join(output_experiment_path_all_comparison, 'data', main_folder_id, 'pair_{}_type_{}_is_dff_{}.pkl'.format(self.get_astro_pair_id(astro_pair), file_id, dff_mode))
                            csv_corr_path = os.path.join(output_experiment_path_all_comparison, 'plots', main_folder_id + '_csv', 'pair_{}_type_{}_is_dff_{}.csv'.format(self.get_astro_pair_id(astro_pair), file_id, dff_mode))

                            behaviour_list_compare =['rest', 'running', 'stick_rest', 'stick_run_ind_15', 'default']
                            fig, res_d = self.get_compare_states_all_xcorr(astro_pair, align_setting=align_setting, dff_mode=dff_mode, n_fake_samples=1, save_pkl_path=save_pkl_path, filter_duration=filter_duration,
                                                                            behaviour_l=behaviour_list_compare)
                            saving_utils.save_plotly_fig(fig, fig_corr_path)
                            saving_utils.save_csv_dict(res_d, csv_corr_path, key_order=behaviour_list_compare)
        '''


        '''
        print('Correlation plots (rest 0 run 0, rest 1 run 1, random)')
        file_id = 'xcorr_compare_between_states'
        if astroA_l_pairs is not None:
            for dff_mode in [False]:
                #for align_setting in ['xcorr', 'xcorr_free']:
                for align_setting in ['xcorr']:
                    for astro_pair in astroA_l_pairs:
                        #for filter_duration in [[None, None], [None, 1], [1, None]]:
                        for filter_duration in [[None, None]]:
                            main_folder_id = 'correlations_no_align' if align_setting == 'xcorr_free' else 'correlations'
                            if (filter_duration[0] == None and filter_duration[1] == 1):
                                main_folder_id += '_short_events'
                            if (filter_duration[0] == 1 and filter_duration[1] == None):
                                main_folder_id += '_long_events'
                            fig_corr_path = os.path.join(output_experiment_path_all_comparison, 'plots', main_folder_id, 'pair_{}_type_{}_is_dff_{}'.format(self.get_astro_pair_id(astro_pair), file_id, dff_mode))
                            save_pkl_path = os.path.join(output_experiment_path_all_comparison, 'data', main_folder_id, 'pair_{}_type_{}_is_dff_{}.pkl'.format(self.get_astro_pair_id(astro_pair), file_id, dff_mode))
                            csv_corr_path = os.path.join(output_experiment_path_all_comparison, 'plots', main_folder_id + '_csv', 'pair_{}_type_{}_is_dff_{}.csv'.format(self.get_astro_pair_id(astro_pair), file_id, dff_mode))
                            fig, res_d = self.get_compare_states_same_astro_all_xcorr(astro_pair, align_setting=align_setting, dff_mode=dff_mode, n_fake_samples=100, save_pkl_path=save_pkl_path, filter_duration=filter_duration)

                            print('RES D', res_d)
                            saving_utils.save_plotly_fig(fig, fig_corr_path)
                            saving_utils.save_csv_dict(res_d, csv_corr_path, key_order=list(res_d.keys()))

        '''
        #TODO RUUN THESE AGAIN


        """
        #USING GOOD PAIRS FROM HERE ON
        #RUN AFTER
        file_id = 'xcorr_compare_between_group'
        if astroA_l_good_pairs is not None:
            for dff_mode in [False]:
                #for align_setting in ['xcorr', 'xcorr_free']:
                #NOT USING ALIGN SETTING
                for align_setting in ['xcorr']:
                    #for filter_duration in [[None, None], [None, 1], [1, None]]:
                    #for filter_duration in [[None, None], [None, 1], [1, None]]:
                    for filter_duration in [[None, None]]:
                        main_folder_id = 'correlations_no_align' if align_setting == 'xcorr_free' else 'correlations'
                        if (filter_duration[0] == None and filter_duration[1] == 1):
                            main_folder_id += '_short_events'
                        if (filter_duration[0] == 1 and filter_duration[1] == None):
                            main_folder_id += '_long_events'

                        fig_corr_path = os.path.join(output_experiment_path_all_comparison, 'plots', main_folder_id, 'type_{}_is_dff_{}'.format(file_id, dff_mode))
                        save_pkl_path = os.path.join(output_experiment_path_all_comparison, 'data', main_folder_id, 'type_{}_is_dff_{}.pkl'.format(file_id, dff_mode))
                        csv_corr_path = os.path.join(output_experiment_path_all_comparison, 'plots', main_folder_id + '_csv', 'type_{}_is_dff_{}.csv'.format(file_id, dff_mode))

                        fig, res_d = self.get_compare_between_group_xcorr(astroA_l_good_pairs, dff_mode=dff_mode, n_fake_samples=5, save_pkl_path=save_pkl_path, filter_duration=filter_duration)

                        saving_utils.save_plotly_fig(fig, fig_corr_path)
                        saving_utils.save_csv_dict(res_d, csv_corr_path, key_order=list(res_d.keys()))
        """


        """
        save_folder = os.path.join(output_experiment_path_all_comparison, 'data', 'control')
        plot_folder = os.path.join(output_experiment_path_all_comparison, 'plots', 'control')

        print('CONTROLS plot')


        print('Recombination results...')
        save_recombination_pkl_path = os.path.join(save_folder, 'recombination.pkl')
        fig, res_d = self.get_compare_between_group_xcorr(astroA_l_good_pairs, dff_mode=False, n_fake_samples=1, save_pkl_path=save_recombination_pkl_path)
        recombination_corrs = res_d['between']
        recombination_rand_corrs = res_d['random']

        print('Recombination CORRS', recombination_corrs)
        print('Recombination rand corrs', recombination_rand_corrs)

        #between_id
        #between
        #random
        print('Random sample results...')
        save_random_pair_pkl_path = os.path.join(save_folder, 'random_pair.pkl')
        if os.path.isfile(save_random_pair_pkl_path):
            print('FILE EXISTS', save_random_pair_pkl_path)
            random_pair_corrs = saving_utils.load_pickle(save_random_pair_pkl_path)
        else:
            random_pair_corrs = []
            for astroA_pair in astroA_l_good_pairs:
                d = compare_astro_utils.alignment_counter(astroA_pair[0], astroA_pair[1],
                                                            n_fake_samples=10,
                                                            align_setting='xcorr',
                                                            eval_setting='xcorr',
                                                            fake_sample_setting='from_astro',
                                                            p=1,
                                                            behaviour='default',
                                                            filter_duration=[None, None],
                                                            with_output_details=True)
                random_pair_corrs.extend(d['num_fake'])
            saving_utils.save_pickle(random_pair_corrs, save_random_pair_pkl_path)
        print('Random pair corrs:', random_pair_corrs)

        print('Flip control results...')
        save_flip_pkl_path = os.path.join(save_folder, 'flip.pkl')
        if os.path.isfile(save_flip_pkl_path):
            print('File exists', save_flip_pkl_path)
            flip_corrs = saving_utils.load_pickle(save_flip_pkl_path)
        else:
            flip_corrs = []
            for astroA in astroA_l_good:
                for num_rot in range(1, 6):
                    astro_grid, _, _,_ = compare_astro_utils.get_filters_compare([astroA], p=1, dff_mode=False, behaviour='default')
                    astro_grid = astro_grid[0]

                    astro_grid_rot_1 = np.copy(astro_grid)
                    astro_grid_border_1 = np.copy(astroA.border)

                    if num_rot < 4:
                        astro_grid_rot_2 = np.rot90(np.copy(astro_grid), k=num_rot)
                        astro_grid_border_2 = np.rot90(np.copy(astroA.border), k=num_rot)
                    elif num_rot == 5:
                        astro_grid_rot_2 = np.flipud(np.copy(astro_grid))
                        astro_grid_border_2 = np.flipud(np.copy(astroA.border))
                    elif num_rot == 6:
                        astro_grid_rot_2 = np.fliplr(np.copy(astro_grid))
                        astro_grid_border_2 = np.fliplr(np.copy(astroA.border))

                    d = compare_astro_utils.alignment_counter(astroA, astroA,
                                                                n_fake_samples=0,
                                                                align_setting='param',
                                                                eval_setting='xcorr',
                                                                fake_sample_setting='from_astro',
                                                                grid_target=astro_grid_rot_1,
                                                                grid_source=astro_grid_rot_2,
                                                                target_border_grid=astro_grid_border_1,
                                                                source_border_grid=astro_grid_border_2,
                                                                move_vector=[0,0],
                                                                p=1,
                                                                behaviour='default',
                                                                with_output_details=True)
                    flip_corrs.append(d['num_compare'])
            saving_utils.save_pickle(flip_corrs, save_flip_pkl_path)
        print('Flip corrs', flip_corrs)

        print('LENS, random pair, flip, recombination')
        print(len(random_pair_corrs), len(flip_corrs), len(recombination_corrs))
        x =['Random simulation', 'Flip Control', 'Recombination Control']
        y = [random_pair_corrs, flip_corrs, recombination_corrs]

        fig = plotly_utils.plot_point_box_revised(x, y, title='Mean +/- standard deviation of controls', x_title='', y_title='xcorr', err_type='std')
        saving_utils.save_plotly_fig(fig, os.path.join(plot_folder, 'control_plot'))
        """

        '''
        plt.ioff()
        print('Plotting Size vs Time correlation plot...')
        path = os.path.join(output_experiment_path_all_comparison, 'plots', 'size_v_time_corr_ALL')
        areas_all = []
        times_all = []

        for astroA in astroA_l:
            areas_all.extend(np.log(astroA.res_d['area']))
            times_all.extend(astroA.res_d['time_s'])
        areas_all = np.array(areas_all)
        times_all = np.array(times_all)
        r, p = stat_utils.get_pearsonr(times_all, areas_all)

        df = pd.DataFrame({'Size': areas_all, 'Time': times_all})

        title ='Size vs Time correlation plot'
        text = 'r = {}, p < {}'.format(general_utils.truncate(r, 2), p)
        for kind in ['reg', 'hex', 'kde']:
            plotly_utils.seaborn_joint_grid(df, 'Size', 'Time', kind=kind, text=text)
            plt.savefig(os.path.join(path, '{}.svg'.format(kind)))
            plt.savefig(os.path.join(path, '{}.png'.format(kind)))
        '''

        '''
        print('---------------------------------')
        print('EVENTS VS SPEED PLOTS...')
        print('---------------------------------')


        speed_event_tuple_d = {}
        n_bins_l = [3, 5, 10]
        n_frame_splits_l = [15, int(astroA_l[0].minute_frames/6)]

        for eval_type in ['max', 'mean']:
            for n_bins in n_bins_l:
                for n_frame_splits in n_frame_splits_l:
                    path = os.path.join(output_experiment_path_all_comparison, 'plots', 'speed_v_events_ALL', 'eval_type={}_splits={}_bins={}'.format(eval_type, n_frame_splits, n_bins))
                    for astroA in astroA_l:
                            #split n frames. Measure average speed in that bin. Measure how many events in that bin.
                            #add to histogram
                            #10 second frame splits
                            total_frames = len(astroA.indices_d['default'])
                            num_chunks = total_frames//n_frame_splits
                            print('NUM FRAME SPLITS {}, TOTAL FRAMES {} NUM CHUNKS {}'.format(n_frame_splits, total_frames, num_chunks))
                            split_arr_i_l = np.array_split(astroA.indices_d['default'], num_chunks)
                            speed_event_tuple_l = aqua_utils.speed_event_tuple(astroA, split_arr_i_l, num_events_only=True, eval_type=eval_type)

                            speed_event_tuple_d[astroA.print_id] = speed_event_tuple_l

                    #Find maximum speed, for bounds of histogram
                    max_speed = 0
                    for k in speed_event_tuple_d.keys():
                        max_speed_k = np.max(np.array([speed for speed, ev_l in speed_event_tuple_d[k]]))
                        #print('MAX SPEED {} : {}'.format(k, max_speed_k))
                        if max_speed_k > max_speed:
                            max_speed = max_speed_k

                    #print('MAX SPEED' , max_speed)

                    events_bins_d = {}
                    bin_values = np.linspace(0, max_speed, n_bins)

                    for astroA in astroA_l:
                        events_bins = [[] for i in range((n_bins-1))]
                        speed_event_tuple = speed_event_tuple_d[astroA.print_id]

                        for sp_ev_tup in speed_event_tuple:
                            ind = np.searchsorted(bin_values, sp_ev_tup[0], side='right')-1
                            if ind == len(events_bins):
                                ind -= 1
                            events_bins[ind].append(sp_ev_tup[1] / n_frame_splits)

                        #events_bins_avg = [np.mean(events_bins[i]) for i in range(len(events_bins))]

                        events_bins_d[astroA.print_id] = events_bins

                    x = bin_values[:-1]
                    names_l = list(events_bins_d.keys())

                    x_l = [x for i in range(len(astroA_l))]
                    y_l_l = [events_bins_d[k] for k in names_l]
                    x_l_dpl = [tup[0] for tup in speed_event_tuple]
                    y_l_dpl = [tup[1] for tup in speed_event_tuple]
                    r, p = stat_utils.get_pearsonr(y_l_dpl, x_l_dpl)
                    df = pd.DataFrame({'Events': y_l_dpl, 'Speed': x_l_dpl})

                    fig, stats_d = plotly_utils.plot_scatter_mult_with_avg(x, y_l_l, None, names_l, mode='lines', title='scatter', x_title='Speed (cm/s)', y_title='',
                                        xrange=None, yrange=None, confidence=True, with_stats=True, point_box=False, mean_width_size=5)
                    saving_utils.save_plotly_fig(fig, path)
                    print('KEYS', stats_d.keys())
                    print('THE STTS D X', stats_d['x'])
                    df_data_m = DataFrame(stats_d['mean_l_l'], columns=stats_d['x'], index=stats_d['names'])
                    df_ci = DataFrame(stats_d['conf_95'], columns=stats_d['x'], index=stats_d['names'])
                    df_mean = DataFrame([stats_d['mean'], stats_d['mean_conf']], columns=stats_d['x'], index=['mean', 'conf_95'])
                    df_data_m.to_csv(path + '-data_means.csv')
                    df_ci.to_csv(path + '-data_ci.csv')
                    df_mean.to_csv(path + '-mean_and_CI.csv')

                    title ='Events vs Speed correlation plot'
                    text = 'r = {}, p < {}'.format(general_utils.truncate(r, 2), p)
                    for kind in ['reg', 'hex', 'kde']:
                        plotly_utils.seaborn_joint_grid(df, 'Speed', 'Events', kind=kind, text=text)
                        plt.savefig(path + '_corr_{}.svg'.format(kind))
                        plt.savefig(path + '_corr_{}.png'.format(kind))
            print('---------------------------------')
        '''
        '''
        print('Plotting correlation of splitted plots in 3 parts...')

        save_folder = os.path.join(output_experiment_path_all_comparison, 'data', 'split_correlation_all')
        plot_folder = os.path.join(output_experiment_path_all_comparison, 'plots', 'split_correlation_all')
        save_splits_pkl_path = os.path.join(save_folder, 'between_splits.pkl')
        save_day_splits_pkl_path = os.path.join(save_folder, 'between_days.pkl')
        save_random_pkl_path = os.path.join(save_folder, 'random.pkl')
        save_bh_splits_pkl_path = os.path.join(save_folder, 'between_rest_run.pkl')
        #1 random simulations
        #2 (correlation between splits days with variable the splits (so not between days) 3 split correlations with each other (only day 0 and day 1). day 0 splitted 3 times and correlated between each other. same with day 1
        #3 (correlation between splits days with variable the between days)) the day 0 and day 1 splitted and then compared between each other between days
        #'split_correlation_all'
        #for bh_l in ['default', 'rest', 'running']:
        #4 (correlation between split days with variable the rest-run behaviour)
        for bh in ['rest']:
            #2
            fig, res_splits_l = self.get_between_split_split_xcorr(astroA_long_l, bh=bh, save_pkl_path=save_splits_pkl_path)
            #3
            fig_2, res_day_splits_l = self.get_between_day_split_xcorr(day_0_1_pairs, bh=bh, save_pkl_path=save_day_splits_pkl_path)
            #4
            fig_3, res_bh_splits_l = self.get_between_bh_split_xcorr(astroA_long_l, bh_pair=['rest','running'], save_pkl_path=save_bh_splits_pkl_path)
            #1
            if os.path.isfile(save_random_pkl_path):
                print('FILE EXISTS')
                random_l = saving_utils.load_pickle(save_random_pkl_path)
            else:
                random_l = []
                for astroA in astroA_long_l:
                    random_l.extend(self.get_random_corrs_self(astroA, bh, n_fake_samples=3))
            if save_random_pkl_path is not None:
                saving_utils.save_pickle(random_l, save_random_pkl_path)

            x = ['Random', 'Self splits', 'Rest-Run splits', 'Day 0-1 Splits']
            y = [random_l, res_splits_l, res_bh_splits_l, res_day_splits_l]

            print('LENS', [len(y_i) for y_i in y])
            fig, stats_d = plotly_utils.plot_point_box_revised(x, y, title='Split correlations (between splits)- {}'.format(bh), x_title='', y_title='Xcorr value', with_stats=True)
            saving_utils.save_plotly_fig(fig, os.path.join(plot_folder, 'splits'))
            saving_utils.save_csv_dict(stats_d, os.path.join(plot_folder, 'splits' + '.csv'), key_order=['x', 'mean', 'conf_95'])

            results_dict = {x[i] : y[i] for i in range(len(x))}
            results_dict['x'] = x
            key_order = ['x']
            key_order.extend(x)
            saving_utils.save_csv_dict(results_dict, os.path.join(plot_folder, 'splits_data' + '.csv'), key_order=key_order)

            return fig
            '''

    def get_random_corrs_self(self, astroA, bh, n_fake_samples=3):
        random_l = []
        d = compare_astro_utils.alignment_counter(astroA, astroA,
                                                    n_fake_samples=n_fake_samples,
                                                    align_setting='param',
                                                    eval_setting='xcorr',
                                                    fake_sample_setting='from_astro',
                                                    move_vector=[0, 0],
                                                    p=1,
                                                    behaviour=bh)
        return d['num_fake']

    def get_between_bh_split_xcorr(self, astroA_l, bh_pair=['rest', 'running'], n_chunks=3, dff_mode=False, save_pkl_path=None, filter_duration=(None, None)):
        '''
        Split bh_pair[0] into 3 splits. Correlate with whole of bh_pair[1]
        '''
        if os.path.isfile(save_pkl_path):
            print('FILE EXISTS')
            res_l = saving_utils.load_pickle(save_pkl_path)
        else:
            event_grid_splits_d = {}
            astros_d = {}
            for astroA in astroA_l:
                print(astroA.print_id)
                event_grid_splits_d[astroA.print_id] = aqua_utils.split_n_event_grids(astroA, bh=bh_pair[0], n=n_chunks)
                astros_d[astroA.print_id] = astroA

            res_l = []
            for astroA_k in event_grid_splits_d.keys():
                #Get correlations of splits between splits same days
                astroA_splits_l = event_grid_splits_d[astroA_k]
                bh_split = astros_d[astroA_k].event_grids_1min[bh_pair[1]]
                for i in range(n_chunks):
                    split_i = astroA_splits_l[i]

                    d = compare_astro_utils.alignment_counter(astros_d[astroA_k], astros_d[astroA_k],
                                                                n_fake_samples=0,
                                                                align_setting='param',
                                                                eval_setting='xcorr',
                                                                fake_sample_setting='from_astro',
                                                                grid_target=bh_split,
                                                                grid_source=split_i,
                                                                move_vector=[0, 0],
                                                                p=1,
                                                                behaviour=bh_pair[0],
                                                                filter_duration=filter_duration,
                                                                with_output_details=True)
                    res_l.append(d['num_compare'])
        if save_pkl_path is not None:
            saving_utils.save_pickle(res_l, save_pkl_path)

        x = ['Split correlations']
        y = [np.copy(np.array(res_l))]
        print('THE Y', y)
        fig = plotly_utils.plot_point_box_revised(x, y, title='{} Split correlations (between splits)- {}'.format(n_chunks, '_'.join(bh_pair)), x_title='', y_title='Xcorr value')
        return fig, res_l


    def get_between_split_split_xcorr(self, astroA_l, bh='default', n_chunks=3, dff_mode=False, save_pkl_path=None, filter_duration=(None, None)):
        if os.path.isfile(save_pkl_path):
            print('FILE EXISTS')
            res_l = saving_utils.load_pickle(save_pkl_path)
        else:
            event_grid_splits_d = {}
            astros_d = {}
            for astroA in astroA_l:
                print(astroA.print_id)
                event_grid_splits_d[astroA.print_id] = aqua_utils.split_n_event_grids(astroA, bh=bh, n=n_chunks)
                astros_d[astroA.print_id] = astroA

            res_l = []
            for astroA_k in event_grid_splits_d.keys():
                #Get correlations of splits between splits same days
                astroA_splits_l = event_grid_splits_d[astroA_k]

                for i in range(n_chunks):
                    for j in range(i+1, n_chunks):
                        print(i, j)
                        split_i = astroA_splits_l[i]
                        split_j = astroA_splits_l[j]

                        d = compare_astro_utils.alignment_counter(astros_d[astroA_k], astros_d[astroA_k],
                                                                    n_fake_samples=0,
                                                                    align_setting='param',
                                                                    eval_setting='xcorr',
                                                                    fake_sample_setting='from_astro',
                                                                    grid_target=split_i,
                                                                    grid_source=split_j,
                                                                    move_vector=[0, 0],
                                                                    p=1,
                                                                    behaviour=bh,
                                                                    filter_duration=filter_duration,
                                                                    with_output_details=True)
                        res_l.append(d['num_compare'])
        if save_pkl_path is not None:
            saving_utils.save_pickle(res_l, save_pkl_path)

        x = ['Split correlations']
        y = [np.copy(np.array(res_l))]
        print('THE Y', y)
        fig = plotly_utils.plot_point_box_revised(x, y, title='{} Split correlations (between splits)- {}'.format(n_chunks, bh), x_title='', y_title='Xcorr value')
        return fig, res_l


    def get_between_day_split_xcorr(self, astroA_l_pairs, bh='default', n_chunks=3, dff_mode=False, n_fake_samples=5, save_pkl_path=None, filter_duration=(None, None)):
        if os.path.isfile(save_pkl_path):
            print('FILE EXISTS')
            res_l = saving_utils.load_pickle(save_pkl_path)
        else:
            res_l = []

            event_grid_splits_d = {}

            for astro_pair in astroA_l_pairs:
                pair_k = self.get_astro_pair_id(astro_pair)
                event_grid_splits_d[pair_k] = {'day_0' : None, 'day_x' : None}
                #Split each astro into 3
                event_grid_splits_d[pair_k]['day_0'] = aqua_utils.split_n_event_grids(astro_pair[0], bh=bh, n=n_chunks)
                event_grid_splits_d[pair_k]['day_x'] = aqua_utils.split_n_event_grids(astro_pair[1], bh=bh, n=n_chunks)
                event_grid_splits_d[pair_k]['astro_pair'] = astro_pair
            #Get all split correlations between day 0 and day x of same astro pair

                #All possible here (note the 2nd for loop different than function above)
                astro_pair = event_grid_splits_d[pair_k]['astro_pair']
                d_temp = compare_astro_utils.alignment_counter(astro_pair[0], astro_pair[1],
                                                                n_fake_samples=0,
                                                                align_setting='xcorr',
                                                                eval_setting='xcorr',
                                                                fake_sample_setting='from_astro',
                                                                p=1,
                                                                behaviour='default',
                                                                dff_mode=dff_mode)
                move_vector = d_temp['move_vector']
                for i in range(n_chunks):
                    for j in range(n_chunks):
                        print(i, j)
                        split_i = event_grid_splits_d[pair_k]['day_0'][i]
                        split_j = event_grid_splits_d[pair_k]['day_x'][j]

                        d = compare_astro_utils.alignment_counter(astro_pair[0], astro_pair[1],
                                                                    n_fake_samples=0,
                                                                    align_setting='param',
                                                                    eval_setting='xcorr',
                                                                    fake_sample_setting='from_astro',
                                                                    grid_target=split_i,
                                                                    grid_source=split_j,
                                                                    move_vector=move_vector,
                                                                    p=1,
                                                                    behaviour=bh,
                                                                    filter_duration=filter_duration,
                                                                    with_output_details=True)
                        res_l.append(d['num_compare'])

        if save_pkl_path is not None:
            saving_utils.save_pickle(res_l, save_pkl_path)

        x = ['Split correlations']
        y = [np.copy(np.array(res_l))]
        fig = plotly_utils.plot_point_box_revised(x, y, title='{} Split correlations (between days) - {}'.format(n_chunks, bh), x_title='', y_title='Xcorr value')
        return fig, res_l

#--------#--------#--------#--------#--------#--------#--------#--------#--------#--------
    def generate_corr_data(self, astroA):
        output_experiment_path = self.get_output_experiment_path(astroA)
        print('Making dirs', output_experiment_path)
        self.setup_file_folders(output_experiment_path)

        print(output_experiment_path)
        print('Generating fake sample correlations and split correlations...')
        #Will use these to compare how much to split before losing correlation

        for p in self.filter_probs:
            samples_save_path = os.path.join(output_experiment_path, 'files', 'correlations', 'fake_sample_p={}.pkl'.format(p))
            samples_corr_d = correlation_utils.get_corr_astro_samples_v2(astro_xc=astroA, astro_base=astroA, p=p, n_samples=self.n_samples_corr_fake)
            saving_utils.save_pickle(samples_corr_d, samples_save_path)

            #splits_save_path = os.path.join(output_experiment_path, 'files', 'correlations', 'splits_p={}.pkl'.format(p))
            #splits_corr_d = correlation_utils.get_splits_corr(astroA, num_frames_splits_l=self.num_frames_splits_l, p=p, max_comparisons=self.max_split_comparison_samples)
            #saving_utils.save_pickle(splits_corr_d, splits_save_path)

        print('Writing csv...')
        duration_csv_path = os.path.join(output_experiment_path, 'files', 'csv', 'duration_split_ratios.csv')
        self.write_csv_duration_splits(astroA, duration_csv_path)

    def generate_corr_data_pair(self, astroA_l):
        output_experiment_path_comparison, days_str, day_l_s, astroA_l_s = self.setup_comparison_vars(astroA_l)
        print(output_experiment_path_comparison)
        print('Making dirs', output_experiment_path_comparison)
        self.setup_file_folders_comparison(output_experiment_path_comparison)

        for p in self.filter_probs:
            print(p)
            d = {}
            corr_compare_save_path = os.path.join(output_experiment_path_comparison, 'files', 'correlations', 'corr_compare_p={}.pkl'.format(p))

            astro_filt_l, astro_all_filt, astro_nz_bool_l, astro_all_nz_bool = compare_astro_utils.get_filters_compare(astroA_l_s, p=p)

            #1 - self correlation
            corr_res_self, max_corr_self, move_vector_self, max_coord_self = correlation_utils.get_cross_correlation_2D_info_compare(astro_filt_l[0], astro_filt_l[0])
            corr_res, max_corr, move_vector, max_coord = correlation_utils.get_cross_correlation_2D_info_compare(astro_filt_l[0], astro_filt_l[1])

            #3 - astroA - astroB fake sample correlations
            samples_d = correlation_utils.get_corr_astro_samples_v2(astro_xc=astroA_l[0], astro_base=astroA_l[1], p=p, n_samples=self.n_samples_corr_fake)

            d['self'] = {'max_corr' : max_corr_self,
                        ' corr_res' : corr_res_self,
                        'move_vector' : move_vector_self,
                        'max_coord' : max_coord_self }
            d['compare'] = {'max_corr' : max_corr,
                        ' corr_res' : corr_res,
                        'move_vector' : move_vector,
                        'max_coord' : max_coord}
            d['samples'] = samples_d

            saving_utils.save_pickle(d, corr_compare_save_path)

    def parse_prob(self, path):
        base_name = os.path.splitext(os.path.basename(path))[0]
        prob_v = float(base_name.split('=')[-1])
        return prob_v

    def read_corr_pair_data(self, astroA_l):
        output_experiment_path_comparison, days_str, day_l_s, astroA_l_s = self.setup_comparison_vars(astroA_l)
        comparison_paths = glob.glob(os.path.join(output_experiment_path_comparison, 'files/correlations/corr_compare_*.pkl'))

        corr_pair_d = {}
        for comparison_path in comparison_paths:
            prob_k = self.parse_prob(comparison_path)
            print('Prob k', prob_k)
            corr_pair_d[prob_k] = saving_utils.load_pickle(comparison_path)
        return corr_pair_d

    def read_corr_data(self, astroA):
        experiment_path = self.get_output_experiment_path(astroA)
        print('Experiment path', experiment_path)

        fake_sample_corr_paths = glob.glob(os.path.join(experiment_path, 'files/correlations/fake_sample_*.pkl'))
        #splits_corr_paths = glob.glob(os.path.join(experiment_path, 'files/correlations/splits_*.pkl'))

        fake_corr_d = {}
        #splits_corr_d = {}

        for fake_sample_path in fake_sample_corr_paths:
            fake_corr_d[str(self.parse_prob(fake_sample_path))] = saving_utils.load_pickle(fake_sample_path)

        #for split_path in splits_corr_paths:
        #    splits_corr_d[str(self.parse_prob(split_path))] = saving_utils.load_pickle(split_path)

        #return fake_corr_d, splits_corr_d
        return fake_corr_d

    def setup_comparison_vars(self, astroA_l):
        experiment_id_l = []
        day_l = []
        for astroA in astroA_l:
            experiment_id_l.append('/'.join(astroA.experiment_path.split('/')[-3:-1]))
            day_l.append(int(astroA.experiment_path.split('/')[-1].split('_')[-1]))

        if len(set(experiment_id_l)) != 1:
            print('Different experiment ids, stopping', experiment_id_l)
            return
        sort_i = np.argsort(day_l)
        day_l_s = [day_l[i] for i in sort_i]
        astroA_l_s = [astroA_l[i] for i in sort_i]

        days_str = 'days_' + '_'.join([str(day) for day in day_l_s])
        output_experiment_path_comparison = os.path.join(self.output_folder,
                                                        experiment_id_l[0],
                                                        days_str)
        print('done')
        return output_experiment_path_comparison, days_str, day_l_s, astroA_l_s

    def setup_comparison_all_vars(self, astroA_l):
        experiment_id_l = []
        day_l = []
        for astroA in astroA_l:
            experiment_id_l.append('/'.join(astroA.experiment_path.split('/')[-3:-1]))
            day_l.append(int(astroA.experiment_path.split('/')[-1].split('_')[-1]))

        sort_i = np.argsort(day_l)
        day_l_s = [day_l[i] for i in sort_i]
        astroA_l_s = [astroA_l[i] for i in sort_i]

        days_str = 'days_' + '_'.join([str(day) for day in day_l_s])
        output_experiment_path_all_comparison = os.path.join(self.output_folder, 'astro_only', 'all')

        print('done')
        return output_experiment_path_all_comparison, days_str, day_l_s, astroA_l_s


    def get_behaviour_basic_plots(self, astroA):
        figs = {}

        figs['stick_bin'] = plotly_utils.plot_scatter_fmt(x=np.arange(len(astroA.stick_bin)), y=astroA.stick_bin, astype='int', straight_lines_only=True, title='Stick', x_title='Frame', y_title='Off whisker/On whisker')
        figs['speed_bin'] = plotly_utils.plot_scatter_fmt(x=np.arange(len(astroA.speed_bin)), y=astroA.speed_bin, astype='int', straight_lines_only=True, title='Speed', x_title='Frame', y_title='Rest/Running')
        figs['whisker_bin'] = plotly_utils.plot_scatter_fmt(x=np.arange(len(astroA.whisker_bin)), y=astroA.whisker_bin, astype='int', straight_lines_only=True, title='Whisker', x_title='Frame', y_title='No whisker/Whisker movement')
        figs['pupil'] = plotly_utils.plot_scatter_fmt(x=np.arange(len(astroA.pupil_values)), y=astroA.pupil_values, astype='float', straight_lines_only=True, title='Pupil', x_title='Frame', y_title='Pupil value')

        figs['stick'] = plotly_utils.plot_scatter(x=np.arange(len(astroA.roi_dict['extra']['stick'])), y=astroA.roi_dict['extra']['stick'], title='Stick', x_title='Frame', y_title='Stick value')
        figs['speed'] = plotly_utils.plot_scatter(x=np.arange(len(astroA.roi_dict['extra']['speed'])), y=astroA.roi_dict['extra']['speed'], title='Speed', x_title='Frame', y_title='Speed value')
        figs['whisker'] = plotly_utils.plot_scatter(x=np.arange(len(astroA.roi_dict['extra']['whiskers'])), y=astroA.roi_dict['extra']['whiskers'], title='Whisker', x_title='Frame', y_title='Whisker value')

        def make_arr(inds, arr_length):
            arr = np.zeros([arr_length])
            arr[inds] = 1
            return arr

        arr_length = len(astroA.stick_bin)

        for k in astroA.indices_d.keys():
            arr = make_arr(astroA.indices_d[k], arr_length)
            figs[k] = plotly_utils.plot_scatter_fmt(x=np.arange(len(arr)), y=arr, title=k, astype='int', straight_lines_only=True, x_title='Frame', y_title='Value')
        return figs

    def get_signal_durations_plot(self, astroA):
        signal_duration_figs = {}
        #Signal durations
        for k in astroA.event_subsets.keys():
            signal_duration_figs[k] = plotly_utils.plot_histogram(astroA.all_durations_d[k], title=' Signal durations histogram ({})'.format(k))
        return signal_duration_figs

    def get_border_plot(self, astroA):
        return plotly_utils.plot_contour(astroA.res_d['border_mask'] + astroA.res_d['clandmark_mask'], title='border_and_landmark_mask', height=600, width=800)

    def get_behaviour_contour_plots(self, astroA):
        '''
        Use 1 min normalized plots
        '''

        fig_heatmap_grids = {}
        fig_heatmap_dff_grids = {}

        print('EVENT GRIDS?' ,astroA.event_grids_1min)
        print('KEYS??', astroA.indices_d.keys())
        #fig_heatmap_dff_grids
        for k in astroA.event_subsets.keys():
            fig_heatmap_grids[k] = plotly_utils.plot_contour(astroA.event_grids_1min[k], title=k + '_event grid', height=600, width=800)

        for k in astroA.event_subsets.keys():
            fig_heatmap_dff_grids[k] = plotly_utils.plot_contour(astroA.event_grids_1min_dff[k], title=k+'_event grid dff', height=600, width=800)

        return fig_heatmap_grids, fig_heatmap_dff_grids

    def get_behaviour_activity_plot(self, astroA):
        activity_ratio_k = np.array(self.filter_keys(astroA))
        activity_ratio_l = np.array([astroA.activity_ratios[k] for k in activity_ratio_k])
        text_values =  np.array(['Frames: ' + str(len(astroA.indices_d[k])) for k in activity_ratio_k])
        activity_i = np.argsort(activity_ratio_l)

        activity_ratio_k_s = activity_ratio_k[activity_i]
        activity_ratio_l_s = activity_ratio_l[activity_i]
        text_values_s = text_values[activity_i]

        activity_ratio_k_s[np.where(activity_ratio_k_s == 'default')] = 'all'
        fig = plotly_utils.plot_bar(x=activity_ratio_k_s, y=activity_ratio_l_s, text_values=['']*len(activity_ratio_l_s), text_size=20, title='Activity ratio (events per voxel)', x_title='', y_title='Events per voxel (%)', margin_b=150)
        plotly_utils.apply_fun_axis_fig(fig, lambda x : x * 100, axis='y',)
        return fig


    def get_behaviour_activity_bar_plot_all(self, astroA_l, bh_l, with_stats=False):
        activity_ratios_np = np.zeros(len(bh_l))
        activity_ratios_num_added = np.zeros(len(bh_l))
        for i, bh_k in enumerate(bh_l):
            for astroA in astroA_l:
                if bh_k in astroA.activity_ratios.keys():
                    activity_ratios_np[i] += astroA.activity_ratios[bh_k]
                    activity_ratios_num_added[i] += 1

        activity_ratios_np /= activity_ratios_num_added
        activity_i = np.argsort(activity_ratios_np)
        activity_ratio_k_s = np.array(bh_l)[activity_i]
        activity_ratio_l_s = activity_ratios_np[activity_i]

        activity_ratio_k_s[np.where(activity_ratio_k_s == 'default')] = 'all'

        fig = plotly_utils.plot_bar(x=activity_ratio_k_s, y=activity_ratio_l_s, text_values=['']*len(activity_ratio_l_s), text_size=20,
                                        title='Activity ratio (events per voxel)', x_title='', y_title='Events per voxel (%)',
                                        margin_b=150,
                                        err_y=[], err_symmetric=None)
        plotly_utils.apply_fun_axis_fig(fig, lambda x : x * 100, axis='y',)


        if with_stats:
            #data = {k : areas[i] for i, k in enumerate(area_keys_s)}
            return fig, {}
        return fig

    def get_behaviour_activity_dot_plot_all(self, astroA_l, bh_l, lines=False):
        activity_ratio_l = []

        for bh in bh_l:
            activity_bh_l = []
            for i, astroA in enumerate(astroA_l):
                if bh in astroA.activity_ratios.keys():
                    activity_bh_l.append(astroA.activity_ratios[bh])
            activity_ratio_l.append(activity_bh_l)

        activity_means = [np.mean(activity_ratios) for activity_ratios in activity_ratio_l]
        print('ACTIVITY MEANS 1', activity_means, 'DONE')
        activity_i = np.argsort(activity_means)

        x = np.array(bh_l)[activity_i]

        y = []
        for i in activity_i:
            y.append(activity_ratio_l[i])

        fig, stats_d = plotly_utils.plot_point_box_revised(x, y, title='Activity ratio', x_title='', y_title='Events per voxel (%)', lines=lines, with_stats=True)
        return fig, stats_d


    def get_behaviour_activity_number_bar_plot_all(self, astroA_l, bh_l, with_stats=False):
        activity_num_np = np.zeros(len(bh_l))
        activity_num_added = np.zeros(len(bh_l))

        for astroA in astroA_l:
            for i, bh_k in enumerate(bh_l):
                if bh_k in astroA.activity_ratios.keys():
                    activity_num_np[i] += (len(astroA.res_d['area'][astroA.event_subsets[bh_k]]) / len(astroA.indices_d[bh_k])) * astroA.minute_frames
                    activity_num_added[i] += 1

        activity_num_np /= activity_num_added
        print('Activity sum: ', activity_num_np)
        activity_i = np.argsort(activity_num_np)

        activity_num_k_s = np.array(bh_l)[activity_i]
        activity_num_l_s = activity_num_np[activity_i]

        activity_num_k_s[np.where(activity_num_k_s == 'default')] = 'all'

        fig = plotly_utils.plot_bar(x=activity_num_k_s, y=activity_num_l_s, text_values=['']*len(activity_num_l_s),
                                    text_size=20, title='Activity number',
                                    x_title='', y_title='Events per minute in state', margin_b=150,
                                    err_y=[], err_symmetric=None)
        if with_stats:
            #data = {k : areas[i] for i, k in enumerate(area_keys_s)}
            return fig, {}

        return fig


    def get_behaviour_activity_number_dot_plot_all(self, astroA_l, bh_l, with_stats=False, lines=False):
        activity_num_l = []

        for bh in bh_l:
            activity_bh_l = []
            for i, astroA in enumerate(astroA_l):
                if bh in astroA.event_subsets.keys():
                    num_events = len(astroA.res_d['area'][astroA.event_subsets[bh]])
                    num_frames = len(astroA.indices_d[bh])
                    activity_bh_l.append((num_events / num_frames) * astroA.minute_frames)

            activity_num_l.append(activity_bh_l)

        activity_means = [np.mean(activity_nums) for activity_nums in activity_num_l]
        print('Activity MEANS HERE: ', activity_means)
        activity_i = np.argsort(activity_means)

        x = np.array(bh_l)[activity_i]

        y = []
        for i in activity_i:
            y.append(activity_num_l[i])

        fig, stats_d = plotly_utils.plot_point_box_revised(x, y, title='Activity number', x_title='', y_title='Events per minute in state', lines=lines, with_stats=True)
        return fig, stats_d

    def get_common_keys(self, astroA_l, bh_l):
        s = set(bh_l)

        for astroA in astroA_l:
            s &= set(astroA.indices_d.keys())

        return np.sort(list(s))

    def get_all_signal_attribute_plot(self, astroA_l, bh_l, type_event='area', type_plot='bar',
                                        y_range=None, divide_y=1, title='', x_title='', y_title='',
                                        error_type='std', err_symmetric=True, with_stats=False):
        areas = [[] for i in range(len(bh_l))]

        for astroA in astroA_l:
            for i, k in enumerate(bh_l):
                if k in astroA.event_subsets.keys():
                    areas_k = astroA.res_d[type_event][astroA.event_subsets[k]]
                    areas[i].extend(areas_k)

        areas_std = np.array([np.std(v_l) for v_l in areas])
        areas_mean = np.array([np.mean(v_l) for v_l in areas])
        areas_conf = []
        for v_l in areas:
            m, l, h = stat_utils.mean_confidence_interval(v_l, confidence=0.95)
            areas_conf.append(m-l)
        areas_conf = np.array(areas_conf)

        areas_i = np.argsort(areas_mean)

        area_keys_s = np.array(bh_l)[areas_i]
        areas_s = np.array(areas)[areas_i]
        areas_mean_s = np.array(areas_mean)[areas_i]
        areas_std_s = np.array(areas_std)[areas_i]
        areas_conf_s = np.array(areas_conf)[areas_i]

        if type_plot == 'bar':
            if error_type == 'std':
                fig = plotly_utils.plot_bar(x=area_keys_s, y=areas_mean_s, text_values=[], text_size=20, title=title, x_title=x_title, y_title=y_title, margin_b=150, err_y=areas_std_s, err_symmetric=err_symmetric)
            elif error_type == 'conf':
                fig = plotly_utils.plot_bar(x=area_keys_s, y=areas_mean_s, text_values=[], text_size=20, title=title, x_title=x_title, y_title=y_title, margin_b=150, err_y=areas_conf_s, err_symmetric=err_symmetric)
        elif type_plot == 'dot':
            fig = plotly_utils.plot_point_box_revised(x=area_keys_s, y=areas_s, title=title, x_title=x_title, y_title=y_title, margin_b=150, y_range=y_range)
        else:
            return None

        if with_stats:
            data = {k : areas_s[i] for i, k in enumerate(area_keys_s)}
            return fig, {'behaviour' : area_keys_s, 'mean' : areas_mean_s, 'std': areas_std_s, 'conf_95': areas_conf_s, 'data' : data}
        return fig

    def get_behaviour_area_plot(self, astroA):
        area_keys = np.array(self.filter_keys(astroA))
        area_l_mean = []
        area_l_std = []
        for k in area_keys:
            area_k = astroA.res_d['area'][astroA.event_subsets[k]]
            area_l_mean.append(np.mean(area_k))
            area_l_std.append(np.std(area_k))

        area_l_mean = np.array(area_l_mean)
        area_l_std = np.array(area_l_std)

        areas_i = np.argsort(area_l_mean)

        area_keys_s = area_keys[areas_i]
        area_l_mean_s = area_l_mean[areas_i]
        area_l_std_s = area_l_std[areas_i]

        fig = plotly_utils.plot_bar(x=area_keys_s, y=area_l_mean_s, text_values=[], text_size=20, title='Sizes of events', x_title='', y_title='Event sizes (\u03bcm<sup>2</sup>)', margin_b=150)
        return fig

    def get_behaviour_amplitude_bar_plot(self, astroA):
        am_keys = np.array(self.filter_keys(astroA))
        am_l_mean = []
        for k in am_keys:
            dff_res = astroA.res_d['dffMax2'][astroA.event_subsets[k]]
            am_l_mean.append(np.mean(dff_res))

        am_l_mean = np.array(am_l_mean)
        am_i = np.argsort(am_l_mean)

        am_keys_s = am_keys[am_i]
        am_l_mean_s= am_l_mean[am_i]

        fig = plotly_utils.plot_bar(x=am_keys_s, y=am_l_mean_s, text_values=[], text_size=20, title='Amplitude (df/f) of events', x_title='', y_title='df/f', margin_b=150)
        return fig

    def get_waterfall_delays_plot_all(self, astroA, return_results_only=False):
        #Unique, no unique
        #Num stick start non num stick start
        #Half second non half second
        #unique_args = [True, False]
        unique_args = [True, False]
        max_duration_args = [None, astroA.duration_small]
        with_stick_num_args = [True]

        figs = {}
        figs_interp = {}

        stick_id = 'stick_exact_start'
        running_id = 'running_exact'
        rest_id = 'rest_exact'

        stick_v_l_d = {}
        running_v_l_d = {}
        no_running_v_l_d = {}

        for un in unique_args:
            for max_duration in max_duration_args:
                for with_stick_num in with_stick_num_args:
                    delay_info_args = {'event_inds_subset' : astroA.event_subsets['default'],
                                       'min_delay' : -20,
                                       'max_delay' : 50,
                                       'max_duration' : max_duration,
                                       'unique_events' : un
                                    }
                    plot_id = '{}-{}-{}'.format('unique' if un else 'notunique',
                                                'max_duration_None' if (max_duration is None) else 'max_duration_' + str(max_duration),
                                                'stick_num_' + str(with_stick_num))

                    if with_stick_num:
                        rand_running = np.random.choice(list(set(astroA.indices_d[running_id]) - set(astroA.indices_d[stick_id])), size=len(astroA.indices_d[stick_id]), replace=False)
                        rand_no_running = np.random.choice(list(set(astroA.indices_d[rest_id]) - set(astroA.indices_d[stick_id])), size=len(astroA.indices_d[stick_id]), replace=False)
                    else:
                        rand_running = list(set(astroA.indices_d[running_id]) - set(astroA.indices_d[stick_id]))
                        rand_no_running = list(set(astroA.indices_d[rest_id]) - set(astroA.indices_d[stick_id]))

                    signal_delays_stick_np, peak_delays_stick_np = aqua_utils.get_delay_info_from_res(astroA.indices_d[stick_id], astroA.res_d, **delay_info_args)
                    signal_delays_running_np, peak_delays_running_np = aqua_utils.get_delay_info_from_res(rand_running, astroA.res_d, **delay_info_args)
                    signal_delays_no_running_np, peak_delays_no_running_np = aqua_utils.get_delay_info_from_res(rand_no_running, astroA.res_d, **delay_info_args)

                    stick_v = np.sort(signal_delays_stick_np)
                    running_v = np.sort(signal_delays_running_np)
                    no_running_v = np.sort(signal_delays_no_running_np)

                    stick_v_l_d[plot_id] = stick_v
                    running_v_l_d[plot_id] = running_v
                    no_running_v_l_d[plot_id] = no_running_v

                    figs[plot_id] = plotly_utils.plot_waterfall(arrays_l=[stick_v, running_v, no_running_v], legend_names=['stick', 'running', 'rest'], title='Signal (event) delays after behaviour', x_title='Delay (s)', y_title='Event id')
                    plotly_utils.apply_fun_axis_fig(figs[plot_id], lambda x : x / astroA.fr, axis='x')
                    figs_interp[plot_id] = plotly_utils.plot_waterfall_interpolate(arrays_l=[stick_v, running_v, no_running_v], legend_names=['stick', 'running', 'rest'], title='Signal (event) delays after behaviour (scaled)', x_title='Delay (s)', y_title='Event id')
                    plotly_utils.apply_fun_axis_fig(figs_interp[plot_id], lambda x : x / astroA.fr, axis='x')

        if return_results_only:
            return [stick_v_l_d, running_v_l_d, no_running_v_l_d]

        return figs, figs_interp

    def get_waterfall_delays_plot_all_mult(self, astroA_l):
        figs_d = {}
        figs_interp_d = {}

        stick_v_l_d = {}
        running_v_l_d = {}
        no_running_v_l_d = {}

        for astroA_i, astroA in enumerate(astroA_l):
            stick_d, running_d, no_running_d = self.get_waterfall_delays_plot_all(astroA, return_results_only=True)
            if astroA_i == 0:
                stick_v_l_d = stick_d
                running_v_l_d = running_d
                no_running_v_l_d = no_running_d


                k_0 = list(stick_d.keys())[0]
                print(k_0)
                print('id {} NUM STICK: {}'.format(astroA.id, len(stick_d[k_0])))

                arrs = [stick_v_l_d, running_v_l_d, no_running_v_l_d]
                for k in stick_d.keys():
                    for arr in arrs:
                        arr[k] = list(arr[k])
            else:
                k_0 = list(stick_d.keys())[0]
                print(k_0)
                print('id {} NUM STICK: {}'.format(astroA.id, len(stick_d[k_0])))
                for k in stick_d.keys():
                    stick_v_l_d[k].extend(stick_d[k])
                    running_v_l_d[k].extend(running_d[k])
                    no_running_v_l_d[k].extend(no_running_d[k])

        for k in stick_v_l_d.keys():
            stick_v = np.sort(stick_v_l_d[k])
            running_v = np.sort(running_v_l_d[k])
            no_running_v = np.sort(no_running_v_l_d[k])

            fig = plotly_utils.plot_waterfall(arrays_l=[stick_v, running_v, no_running_v], legend_names=['stick', 'running', 'rest'], title='Signal (event) delays after behaviour', x_title='Delay (s)', y_title='Event id')

            plotly_utils.apply_fun_axis_fig(fig, lambda x : x / astroA_l[0].fr, axis='x')
            fig_interp = plotly_utils.plot_waterfall_interpolate(arrays_l=[stick_v, running_v, no_running_v], legend_names=['stick', 'running', 'rest'], title='Signal (event) delays after behaviour (scaled) All axons', x_title='Delay (s)', y_title='Event id')
            plotly_utils.apply_fun_axis_fig(fig_interp, lambda x : x / astroA_l[0].fr, axis='x')

            figs_d[k] = fig
            figs_interp_d[k] = fig_interp
        return figs_d, figs_interp_d

    def get_transition_proportion_delays_plot_all(self, astroA_l, before_bh, inds_bh, after_bh,
                                                before_range=20, after_range=50, avg_proportions=False,
                                                delay_step_size=1):
        '''
        inds: the inds i to check
        before_bh: for each i, make sure bh before is before_bh otherwize don't include i
        after_bh: for each i, make sure bh after is after_bh otherwize don't include i
        before_range: the range we look for events
        after_range: the range we look for events
        '''
        #Unique, no unique
        #Num stick start non num stick start
        unique_args = [True, False]
        max_duration_args = [None, astroA_l[0].duration_small]

        figs = {}

        for max_duration in max_duration_args:
            #STICK
            for un in unique_args:
                plot_id = 'prop-{}-{}'.format('unique' if un else 'notunique', 'max_duration_None' if (max_duration is None) else 'max_duration_' + str(max_duration))
                #print('??', after_range)
                #print('???', before_range)
                prop = np.zeros([(after_range+before_range+1)])
                signal_delays_all_l = []

                for astroA in astroA_l:
                    inds = astroA.indices_d[inds_bh]
                    #Filter indices
                    indices_filt_before = aqua_utils.filter_range_inds(inds, astroA.indices_d[before_bh], range=(-before_range, -1), prop=1.0)
                    indices_filt_after = aqua_utils.filter_range_inds(inds, astroA.indices_d[after_bh], range=(1, after_range), prop=1.0)
                    indices_filt = np.array(np.sort(list(set(indices_filt_before) & set(indices_filt_after))))

                    #print('TOTAL IND {} BEFORE {} AFTER {} JOIN {}'.format(len(inds), len(indices_filt_before), len(indices_filt_after), len(indices_filt)))
                    #print(astroA.print_id)
                    #print(indices_filt[0:10])
                    if len(indices_filt) == 0:
                        continue
                    #print('Len indices {} len filt before {} len filt after {} len filt {}'.format(len(inds), len(indices_filt_before), len(indices_filt_after), len(indices_filt)))

                    delay_info_args = {'event_inds_subset' : astroA.event_subsets['default'],
                                   'min_delay' : -before_range,
                                   'max_delay' : after_range,
                                   'max_duration' : max_duration,
                                   'unique_events' : un
                                   }

                    signal_delays_np, peak_delays_np = aqua_utils.get_delay_info_from_res(indices_filt, astroA.res_d, **delay_info_args)
                    signal_delays_all_l.extend(list(signal_delays_np))

                signal_delays_all = np.array(signal_delays_all_l)
                print('Total signals {} {}-{} delay {} {}'.format(len(signal_delays_all), before_bh, after_bh, before_range, after_range))

                for i, delay_x in enumerate(range(-before_range, after_range+1)):
                    if len(signal_delays_all) == 0:
                        prop[i] = 0
                    else:
                        prop[i] = float(np.sum(signal_delays_all == delay_x)) / len(signal_delays_all)

                rem = len(prop) % delay_step_size
                if rem != 0:
                    prop = prop[:-rem]
                prop_step_sum = np.sum(prop.reshape([-1, delay_step_size]), axis=1)

                x_l = [np.arange(-before_range, after_range+1, delay_step_size) for i in range(1)]
                y_l = [prop_step_sum]

                figs[plot_id] = plotly_utils.plot_scatter_mult(x_l, y_l, name_l=['{} to {}'.format(before_bh, after_bh)], mode='lines', title='scatter', x_title='Delay (s)', y_title='Events')
                plotly_utils.apply_fun_axis_fig(figs[plot_id], lambda x : x / astroA.fr, axis='x')
        return figs

    def get_transition_proportion_delays_plot_all_alt(self, astroA_l, before_bh, inds_bh, after_bh,
                                                before_range=20, after_range=50, y_title=None,
                                                delay_step_size=1, fit=False, measure=None, fix_dff_interval=50, confidence=False,
                                                duration_filter=[None, None]):
        '''
        Generate plots of transitions between behaviours lasting for some period of time
        (e.g. 20 frames of rest (before_bh) and then transition to 30 frames of running (after_bh)
        for valid indices in running_start_exact (inds_bh)). We can provide a measure to
        plot a particular measure such as size or amplitude or leave it empty and obtain
        the proportion of events taking place at which delay during these intervals found.

        inds: the inds i to check
        before_bh: for each i, make sure bh before is before_bh otherwize don't include i
        after_bh: for each i, make sure bh after is after_bh otherwize don't include i
        before_range: the range we look for events
        after_range: the range we look for events

        before_delay: the delay of the interval we look for continious befrore and after bh(its actually kind the range...)
        '''

        figs = {}
        signal_delays_all_l_l = []

        if measure is not None:
            event_measure_all_l_l = []

        #DFF max fix, to support both default and the fix
        dff_max_to_fix = (measure == 'dffMax2')
        if measure == 'dffMax2default':
            measure = 'dffMax2'
        #Fix dffMax by adding more range and delay
        if dff_max_to_fix:
            before_range += fix_dff_interval
            after_range += fix_dff_interval

        for astroA in astroA_l:
            inds = astroA.indices_d[inds_bh]
            #Filter indices
            indices_filt_before = aqua_utils.filter_range_inds(inds, astroA.indices_d[before_bh], range=(-before_range, -1), prop=1.0)
            indices_filt_after = aqua_utils.filter_range_inds(inds, astroA.indices_d[after_bh], range=(1, after_range), prop=1.0)
            indices_filt = np.array(np.sort(list(set(indices_filt_before) & set(indices_filt_after))))

            #print('LEN INDICES_FILT: {}'.format(len(indices_filt)))
            #print('TOTAL IND {} BEFORE {} AFTER {} JOIN {}'.format(len(inds), len(indices_filt_before), len(indices_filt_after), len(indices_filt)))
            if len(indices_filt) == 0:
                continue
            #print('Len indices {} len filt before {} len filt after {} len filt {}'.format(len(inds), len(indices_filt_before), len(indices_filt_after), len(indices_filt)))
            #print('LEN INDICES FILT : {}'.format(len(indices_filt)))

            delay_info_args = {'event_inds_subset' : astroA.event_subsets['default'],
                           'min_delay' : -before_range,
                           'max_delay' : after_range,
                           'min_duration' : duration_filter[0],
                           'max_duration' : duration_filter[1],
                           'unique_events' : False,
                           'return_non_unique_delays_arr' : True
                           }

            _, _, _, signal_delays_l_l, peak_mins_l_l, valid_event_i_l_l = aqua_utils.get_delay_info_from_res(indices_filt, astroA.res_d, **delay_info_args)

            #DFF MAX FIX
            #-------------------------------------------------------------------
            print('Num unique indices CHECK {}'.format(len(valid_event_i_l_l)))
            signal_delays_l_l_amended = []
            if measure is not None:
                #Special case for amplitude, we want the exact spot where
                #the maximum takes place, not beginning of event
                #So we increase signal delay by max_frame_i - tBegin to incorporate this
                if dff_max_to_fix:
                    for i, signal_delays_l in enumerate(signal_delays_l_l):
                        valid_event_i_l = valid_event_i_l_l[i]
                        new_delays_l = [(s + astroA.res_d['dffMaxFrame'][valid_event_i_l[j]] - astroA.res_d['tBegin'][valid_event_i_l[j]]) for j, s in enumerate(signal_delays_l)]
                        for j, s in enumerate(new_delays_l):
                            if s > after_range:
                                new_delays_l.pop(j)
                                valid_event_i_l.pop(j)
                        signal_delays_l_l_amended.append(new_delays_l)
                    signal_delays_l_l = signal_delays_l_l_amended
            #-------------------------------------------------------------------

            for i, signal_delays_l in enumerate(signal_delays_l_l):
                signal_delays_all_l_l.append(signal_delays_l)

                if measure is not None:
                    event_measure_all_l_l.append(list(astroA.res_d[measure][valid_event_i_l_l[i]]))

        total_events = np.sum([len(signal_delays_all_l) for signal_delays_all_l in signal_delays_all_l_l])

        #if measure is not None:
        #    total_events2 = np.sum([len(v_l) for v_l in event_measure_all_l_l])
        #print('Total signals {} {}-{} delay {} {}'.format(total_events, before_bh, after_bh, before_range, after_range))
        #Measure or event matrix
        prop_all_np = np.zeros([len(signal_delays_all_l_l), after_range + before_range+1])
        #Count events in case we are using a measure
        ev_count_all_np = np.zeros([len(signal_delays_all_l_l), after_range + before_range+1])

        #Generate individual proportion plots
        for s_i, signal_delays_all_l in enumerate(signal_delays_all_l_l):
            prop = np.zeros([(after_range+before_range+1)])
            ev_count = np.zeros([(after_range+before_range+1)])
            for i, delay_x in enumerate(range(-before_range, after_range+1)):
                if len(signal_delays_all_l) == 0:
                    prop[i] = 0
                else:
                    if measure is None:
                        prop[i] = float(np.sum(np.array(signal_delays_all_l) == delay_x))
                    else:
                        ev_count[i] = float(np.sum(np.array(signal_delays_all_l) == delay_x))
                        valid_delays_i = np.where(np.array(signal_delays_all_l) == delay_x)
                        if len(valid_delays_i[0]) == 0:
                            prop[i] = 0
                        else:
                            prop[i] = np.mean(np.array(event_measure_all_l_l[s_i])[np.where(np.array(signal_delays_all_l) == delay_x)])

            prop_all_np[s_i, :] = prop

            if measure is not None:
                ev_count_all_np[s_i, :] = ev_count

        #Working on proportion plots and event numbers
        if measure is None:
            prop_avg_events = np.sum(prop_all_np, axis=0) / (prop_all_np.shape[0])
            #print('BEFORE EVENTS', np.sum(np.sum(prop_all_np, axis=0)[0:before_range]))
            #print('AFTER EVENTS', np.sum(np.sum(prop_all_np, axis=0)[before_range:]))
            prop_avg_prop = np.sum(prop_all_np, axis=0) / np.sum(prop_all_np)
            prop_total_events = np.sum(prop_all_np, axis=0)
        #Working on durations, areas, amplitudes, we only care about averaging
        #the non-zero values, where there are events.
        else:
            #[num_intervals, interval_size]
            #How many intervals are non zero [1, interval_size]
            count_nz_intervals = np.count_nonzero(ev_count_all_np, axis=0)
            #print('COUNT NZ', count_nz_intervals)
            prop_avg_events = np.sum(prop_all_np, axis=0) / count_nz_intervals
            #Set non existent events to nan, so they aren't showing in the plot
            prop_all_np[ev_count_all_np == 0] = np.nan

        bin_type = 'add' if measure is None else 'mean'
        #TODO HACK
        if delay_step_size != 1 and ((after_range + before_range) // delay_step_size != 2):
            bin_type = 'mean'

        #Fix redo of dffMax to keep original range
        if dff_max_to_fix:
            before_range -= fix_dff_interval
            after_range -= fix_dff_interval

            prop_avg_events = prop_avg_events[fix_dff_interval:-fix_dff_interval]
            prop_all_np = prop_all_np[:, fix_dff_interval:-fix_dff_interval]

        x = np.arange(-before_range, after_range + 1, 1)

        print('CALLING FUNCTION HERE!!')
        fig, bin_stats = plotly_utils.plot_scatter_mult_tree(x=x, y_main=prop_avg_events, y_mult=prop_all_np, mode_main='lines', mode_mult='markers',
                                                    title='Average - Total events: {} Total intervals: {}'.format(total_events, prop_all_np.shape[0]),
                                                    y_title='Num events / interval' if y_title is None else y_title, x_title='Delay (s)', fit=fit, fit_annotation_pos_fix=astroA.fr,
                                                    bin_main_size=delay_step_size, bin_mult_size=delay_step_size, opacity=0.5, confidence=confidence, with_stats=True,
                                                    bin_type=bin_type)
        print('BINS TATS HERE??', bin_stats.keys())
        confidence_format = 'lines' if delay_step_size == 1 else 'bar'
        fig2 = plotly_utils.plot_scatter_mult_tree(x=x, y_main=prop_avg_events, y_mult=prop_all_np, mode_main='lines', mode_mult='markers',
                                                    title='Average - Total events: {} Total intervals: {}'.format(total_events, prop_all_np.shape[0]),
                                                    y_title='Num events / interval' if y_title is None else y_title, x_title='Delay (s)', fit=fit, fit_annotation_pos_fix=astroA.fr,
                                                    bin_main_size=delay_step_size, bin_mult_size=delay_step_size, opacity=0.5, confidence=confidence, y_mult_include=False,
                                                    confidence_format=confidence_format, bin_type=bin_type)


        #Normally we take the mean of the bin. However when we take the number of events in the bin
        #we want to add them up

        if len(x) // delay_step_size > 2:
            plotly_utils.apply_fun_axis_fig(fig, lambda x : x / astroA.fr, axis='x')
            plotly_utils.apply_fun_axis_fig(fig2, lambda x : x / astroA.fr, axis='x')

        #No proportions or total is used if we are doing measure
        if measure is not None:
            return {'event_avg' : fig, 'event_avg_no_mult' : fig2}, bin_stats

        fig3 = plotly_utils.plot_scatter(x=x, y=prop_avg_prop, title='Proportions - plot: Total events: {} Total intervals: {}'.format(total_events, prop_all_np.shape[0]),
                                                y_title='Normalized events (%)' if y_title is None else y_title, x_title='Delay (s)', bin_size=delay_step_size, bin_type=bin_type)

        plotly_utils.apply_fun_axis_fig(fig3, lambda x : x * 100, axis='y')

        if len(x) // delay_step_size > 2:
            plotly_utils.apply_fun_axis_fig(fig3, lambda x : x / astroA.fr, axis='x')

        return {'event_avg' : fig, 'event_avg_no_mult' : fig2,
                    'event_prop' : fig3}, bin_stats

    def get_transition_bh_values_plot_all_alt(self, astroA_l, before_bh, inds_bh, after_bh,
                                                bh_measure='speed',
                                                before_range=20, after_range=50, y_title=None,
                                                delay_step_size=1, fit=False, confidence=False):
        '''
        Get transition plots, but plots the values of behaviours (e.g. speed, stick, ...)
        '''
        figs = {}
        bh_val_all_l = []

        for astroA in astroA_l:
            inds = astroA.indices_d[inds_bh]

            #Filter indices
            indices_filt_before = aqua_utils.filter_range_inds(inds, astroA.indices_d[before_bh], range=(-before_range, -1), prop=1.0)
            indices_filt_after = aqua_utils.filter_range_inds(inds, astroA.indices_d[after_bh], range=(1, after_range), prop=1.0)
            indices_filt = np.array(np.sort(list(set(indices_filt_before) & set(indices_filt_after))))

            if len(indices_filt) == 0:
                continue

            #print('LEN INDICES FILT : {}'.format(len(indices_filt)))
            for center_ind in indices_filt:
                interval_inds = np.arange(center_ind-before_range, center_ind+after_range+1)
                if bh_measure == 'speed':
                    bh_values = astroA.speed_values[interval_inds]
                elif bh_measure == 'pupil':
                    bh_values = astroA.pupil_values[interval_inds]
                else:
                    print('Other measures not supported')
                    return None
                bh_val_all_l.append(bh_values)

        bh_val_all_np = np.zeros([len(bh_val_all_l), before_range + after_range + 1])

        for i, bh_val_l in enumerate(bh_val_all_l):
            if bh_measure == 'speed':
                bh_val_l = np.copy(bh_val_l)
                if before_bh == 'running_semi_exact':
                    bh_val_l[:before_range][bh_val_l[:before_range] == 0] = None
                if after_bh == 'running_semi_exact':
                    bh_val_l[before_range+1:][bh_val_l[before_range+1:] == 0] = None

            bh_val_all_np[i, :] = np.array(bh_val_l)

        bh_val_avg = np.nanmean(bh_val_all_np, axis=0)

        x = np.arange(-before_range, after_range+1, 1)

        fig, bin_stats = plotly_utils.plot_scatter_mult_tree(x=x, y_main=bh_val_avg, y_mult=bh_val_all_np, mode_main='lines', mode_mult='lines',
                                                    title='Total intervals: {}'.format(bh_val_all_np.shape[0]),
                                                    y_title='Speed (cm/s)' if y_title is None else y_title, x_title='Delay (s)', fit=fit, fit_annotation_pos_fix=astroA.fr,
                                                    bin_main_size=delay_step_size, bin_mult_size=delay_step_size, opacity=0.5, confidence=confidence, with_stats=True)
        if len(x) // delay_step_size > 2:
            plotly_utils.apply_fun_axis_fig(fig, lambda x : x / astroA.fr, axis='x')

        confidence_format = 'lines' if delay_step_size == 1 else 'bar'
        if confidence:
            fig2 = plotly_utils.plot_scatter_mult_tree(x=x, y_main=bh_val_avg, y_mult=bh_val_all_np, mode_main='lines', mode_mult='lines',
                                                        title='Total intervals: {}'.format(bh_val_all_np.shape[0]),
                                                        y_title='Speed (cm/s)' if y_title is None else y_title, x_title='Delay (s)', fit=fit, fit_annotation_pos_fix=astroA.fr,
                                                        bin_main_size=delay_step_size, bin_mult_size=delay_step_size, opacity=0.5, confidence=confidence, y_mult_include=False,
                                                        confidence_format=confidence_format)
        if len(x) // delay_step_size > 2:
            plotly_utils.apply_fun_axis_fig(fig2, lambda x : x / astroA.fr, axis='x')

            return {'event_avg' : fig, 'event_avg_no_mult' : fig2}, bin_stats
        else:
            return {'event_avg' : fig}, bin_stats

    def get_transition_outliers_plot(self, astroA_l, before_bh, inds_bh, after_bh,
                                                before_range=20, after_range=50, y_title=None,
                                                delay_step_size=1, fit=False, measure=None, fix_dff_interval=50, confidence=False,
                                                duration_filter=[None, None]):
        '''
        We have 2 behaviours in transition, before bh and after bh.
        We calculate how many events in before bh and after bh are 1,2,3 sd > for measure set
        We then normalize by number of events and also the length of the range of this behaviour

        inds: the inds i to check
        before_bh: for each i, make sure bh before is before_bh otherwize don't include i
        after_bh: for each i, make sure bh after is after_bh otherwize don't include i
        before_range: the range we look for events
        after_range: the range we look for events

        before_delay: the delay of the interval we look for continious befrore and after bh(its actually kind the range...)
        '''
        signal_delays_all_l_l = []
        event_measure_all_l_l = []

        #DFF max fix, to support both default and the fix
        dff_max_to_fix = (measure == 'dffMax2')
        if measure == 'dffMax2default':
            measure = 'dffMax2'
        #Fix dffMax by adding more range and delay
        if dff_max_to_fix:
            before_range += fix_dff_interval
            after_range += fix_dff_interval

        #Events total (all time) and 1,2,3 std thresholds for measure
        all_events = []
        for astroA in astroA_l:
            all_events_individual = astroA.res_d[measure][astroA.event_subsets['default']]
            print('-----', np.mean(all_events_individual))
            all_events.extend(all_events_individual)

        all_events_mean = np.mean(all_events)
        all_events_std = np.std(all_events)
        std_thresholds = np.array([all_events_mean + all_events_std,
                                all_events_mean + 2*all_events_std,
                                all_events_mean + 3*all_events_std])
        print('THRESHOLDS STD', std_thresholds)

        for astroA in astroA_l:
            inds = astroA.indices_d[inds_bh]
            #Filter indices
            indices_filt_before = aqua_utils.filter_range_inds(inds, astroA.indices_d[before_bh], range=(-before_range, -1), prop=1.0)
            indices_filt_after = aqua_utils.filter_range_inds(inds, astroA.indices_d[after_bh], range=(1, after_range), prop=1.0)
            indices_filt = np.array(np.sort(list(set(indices_filt_before) & set(indices_filt_after))))

            #print('LEN INDICES_FILT: {}'.format(len(indices_filt)))
            #print('TOTAL IND {} BEFORE {} AFTER {} JOIN {}'.format(len(inds), len(indices_filt_before), len(indices_filt_after), len(indices_filt)))
            if len(indices_filt) == 0:
                continue
            #print('Len indices {} len filt before {} len filt after {} len filt {}'.format(len(inds), len(indices_filt_before), len(indices_filt_after), len(indices_filt)))
            #print('LEN INDICES FILT : {}'.format(len(indices_filt)))

            delay_info_args = {'event_inds_subset' : astroA.event_subsets['default'],
                           'min_delay' : -before_range,
                           'max_delay' : after_range,
                           'min_duration' : duration_filter[0],
                           'max_duration' : duration_filter[1],
                           'unique_events' : False,
                           'return_non_unique_delays_arr' : True
                           }

            _, _, _, signal_delays_l_l, peak_mins_l_l, valid_event_i_l_l = aqua_utils.get_delay_info_from_res(indices_filt, astroA.res_d, **delay_info_args)

            #DFF MAX FIX
            #-------------------------------------------------------------------
            print('Num unique indices CHECK {}'.format(len(valid_event_i_l_l)))
            signal_delays_l_l_amended = []
            if measure is not None:
                #Special case for amplitude, we want the exact spot where
                #the maximum takes place, not beginning of event
                #So we increase signal delay by max_frame_i - tBegin to incorporate this
                if dff_max_to_fix:
                    for i, signal_delays_l in enumerate(signal_delays_l_l):
                        valid_event_i_l = valid_event_i_l_l[i]
                        new_delays_l = [(s + astroA.res_d['dffMaxFrame'][valid_event_i_l[j]] - astroA.res_d['tBegin'][valid_event_i_l[j]]) for j, s in enumerate(signal_delays_l)]
                        for j, s in enumerate(new_delays_l):
                            if s > after_range:
                                new_delays_l.pop(j)
                                valid_event_i_l.pop(j)
                        signal_delays_l_l_amended.append(new_delays_l)
                    signal_delays_l_l = signal_delays_l_l_amended
            #-------------------------------------------------------------------

            for i, signal_delays_l in enumerate(signal_delays_l_l):
                signal_delays_all_l_l.append(signal_delays_l)

                if measure is not None:
                    event_measure_all_l_l.append(list(astroA.res_d[measure][valid_event_i_l_l[i]]))

        total_events = np.sum([len(signal_delays_all_l) for signal_delays_all_l in signal_delays_all_l_l])

        #if measure is not None:
        #    total_events2 = np.sum([len(v_l) for v_l in event_measure_all_l_l])
        #print('Total signals {} {}-{} delay {} {}'.format(total_events, before_bh, after_bh, before_range, after_range))
        #Measure or event matrix
        prop_all_np = np.zeros([len(signal_delays_all_l_l), after_range + before_range+1])
        #Count events in case we are using a measure
        ev_count_all_np = np.zeros([len(signal_delays_all_l_l), after_range + before_range+1])

        measure_values_all = {'before' : [], 'after' : []}
        for s_i, signal_delays_all_l in enumerate(signal_delays_all_l_l):
            for state in ['before', 'after']:
                if state == 'before':
                    valid_delays_i = np.where(np.array(signal_delays_all_l) < 0)
                elif state == 'after':
                    valid_delays_i = np.where(np.array(signal_delays_all_l) > 0)
                else:
                    print('???')
                    sys.exit()
                measure_values_l = list(np.array(event_measure_all_l_l[s_i])[valid_delays_i])
                measure_values_all[state].extend(measure_values_l)

        measure_values_all['before'] = np.array(measure_values_all['before'])
        measure_values_all['after'] = np.array(measure_values_all['after'])

        name_l = ['1 SD', '2 SD', '3 SD']
        y_l_l = []
        x_l = [['Before', 'After'] for i in range(3)]

        for std_threshold in std_thresholds:
            y_l = [[], []]
            for i, state in enumerate(['before', 'after']):
                sum_t_events = np.sum(measure_values_all[state] > std_threshold)
                norm_t_events = sum_t_events / len(measure_values_all[state])

                print(before_bh, after_bh)
                print('STATE {} SUM T {} NORM T {} ALL MEAS {}'.format(state, sum_t_events, norm_t_events, len(measure_values_all[state])))
                y_l[i].append(norm_t_events)
            y_l_l.append(y_l)
        print(y_l_l)
        fig, stats_d = plotly_utils.plot_scatter_mult(x_l, y_l_l, name_l=name_l, mode='lines+markers', title='scatter', x_title='', y_title='',
                                xrange=None, yrange=None, confidence=False, with_stats=True)
        return fig, stats_d



    def get_proportion_delays_plot_all(self, astroA_l, min_delay=-20, max_delay=50, avg_proportions=False, title=''):
        '''
        For stick find take stick_exact_start when the mouse first hits
        For running and rest:
            Take all rest frames. Stich them and then split into (max_delay-min_delay) frame segments
            Then see the events taking place at each point during the segment from min delay to max delay
        '''
        #Unique, no unique
        #Num stick start non num stick start
        unique_args = [True, False]
        max_duration_args = [None, astroA_l[0].duration_small]

        figs = {}

        stick_id = 'stick_exact_start'
        running_id = 'running_exact'
        rest_id = 'rest_exact'

        #Split into max_delay-min_delay frames
        split_size = (max_delay - min_delay) + 1
        running_prop, rest_prop = self.get_rest_run_proportion_events_interval(astroA_l, running_id='running_exact', rest_id='rest_exact', interval=split_size)

        for max_duration in max_duration_args:
            #STICK
            for un in unique_args:
                plot_id = 'prop-{}-{}'.format('unique' if un else 'notunique', 'max_duration_None' if (max_duration is None) else 'max_duration_' + str(max_duration))

                stick_prop = np.zeros([max_delay-min_delay+1])

                if not avg_proportions:
                    signal_delays_all_l = []

                for astroA in astroA_l:
                    stick_indices_filt = aqua_utils.filter_range_inds(astroA.indices_d[stick_id], astroA.indices_d['running'], range=(min_delay, max_delay), prop=0.95)
                    #print('LEN INDICES PROP: before {} after {}'.format(len(astroA.indices_d[stick_id]), len(stick_indices_filt)))

                    delay_info_args = {'event_inds_subset' : astroA.event_subsets['default'],
                                   'min_delay' : min_delay,
                                   'max_delay' : max_delay,
                                   'max_duration' : max_duration,
                                   'unique_events' : un
                                }
                    signal_delays_stick_np, peak_delays_stick_np = aqua_utils.get_delay_info_from_res(stick_indices_filt, astroA.res_d, **delay_info_args)

                    if avg_proportions:
                        for i, delay_x in enumerate(range(min_delay, max_delay+1)):
                            stick_prop[i] += float(np.sum(signal_delays_stick_np == delay_x)) / len(signal_delays_stick_np)

                    if not avg_proportions:
                        signal_delays_all_l.extend(list(signal_delays_stick_np))

                if avg_proportions:
                    print('SUM OF STICK PROP BEFORE;', np.sum(stick_prop))
                    stick_prop /= len(astroA_l)
                    print('SUM OF STICK PROP:', np.sum(stick_prop))

                if not avg_proportions:
                    signal_delays_all = np.array(signal_delays_all_l)
                    for i, delay_x in enumerate(range(min_delay, max_delay+1)):
                        stick_prop[i] = float(np.sum(signal_delays_all == delay_x)) / len(signal_delays_all)

                x_l = [np.arange(min_delay, max_delay+1) for i in range(3)]
                y_l = [stick_prop, running_prop, rest_prop]

                figs[plot_id] = plotly_utils.plot_scatter_mult(x_l, y_l, name_l=['stick', 'running', 'rest'], mode='lines', title=title, x_title='Delay (s)', y_title='Events')
                plotly_utils.apply_fun_axis_fig(figs[plot_id], lambda x : x / astroA.fr, axis='x')
        return figs

    def get_rest_run_proportion_events_interval(self, astroA_l, running_id='running_exact', rest_id='rest_exact', interval=71):
        running_prop = np.zeros([interval])
        rest_prop = np.zeros([interval])

        for astroA in astroA_l:
            ############################################################
            #RUNNING AND REST
            running_ind = astroA.indices_d[running_id]
            rest_ind = astroA.indices_d[rest_id]

            print('running_ind', running_ind, 'split size' , interval)
            print('running ind now', len(running_ind[:-(len(running_ind) % interval)]))

            if len(running_ind) % interval != 0:
                running_ind = running_ind[:-(len(running_ind) % interval)]

            if len(rest_ind) % interval != 0:
                rest_ind = rest_ind[:-(len(rest_ind) % interval)]

            running_split_l = np.split(running_ind, len(running_ind) / interval)
            rest_split_l = np.split(rest_ind, len(rest_ind) / interval)

            #Add events in delays based on their delay. Ignore events if there is max duration filter

            #For each split of frames, get events in those frames
            split_d = {'default' : astroA.indices_d['default']}
            for i, running_split in enumerate(running_split_l):
                split_d['running_{}'.format(i)] = running_split
            for i, rest_split in enumerate(rest_split_l):
                split_d['rest_{}'.format(i)] = rest_split

            event_subsets, indices_events_bin = aqua_utils.get_event_subsets(split_d, astroA.res_d, after_i=0, before_i=0, to_print=False, return_info=True)

            for k in split_d.keys():
                if k != 'default':
                    #Take indices_d x events and take only current split (split x events)
                    indices_events_k_subset = indices_events_bin[split_d[k], :]
                    #Sum over events to get array of positions where events took place in split_d
                    indices_k_subset_sum = np.sum(indices_events_k_subset, axis=(1))
                    if 'rest' in k:
                        rest_prop += indices_k_subset_sum
                    elif 'running' in k:
                        #Then add these to running prop. At each spot is the number of events that took place
                        running_prop += indices_k_subset_sum
                    else:
                        print('????what', k)
        ############################################################
        running_prop = running_prop / np.sum(running_prop)
        rest_prop = rest_prop / np.sum(rest_prop)
        return running_prop, rest_prop

    '''

    def get_proportion_delays_plot_all_old(self, astroA, min_delay=-20, max_delay=50):
        """
        For stick find take stick_exact_start when the mouse first hits
        For running and rest:
            Take all rest frames. Stich them and then split into (max_delay-min_delay) frame segments
            Then see the events taking place at each point during the segment from min delay to max delay
        """
        #Unique, no unique
        #Num stick start non num stick start
        unique_args = [True, False]
        max_duration_args = [None, astroA.duration_small]
        with_stick_num_args = [True]



        figs = {}
        figs_interp = {}

        stick_id = 'stick_exact_start'





        running_id = 'running_exact'
        rest_id = 'rest_exact'

        for un in unique_args:
            for max_duration in max_duration_args:
                for with_stick_num in with_stick_num_args:
                    delay_info_args = {'event_inds_subset' : astroA.event_subsets['default'],
                                       'min_delay' : min_delay,
                                       'max_delay' : max_delay,
                                       'max_duration' : max_duration,
                                       'unique_events' : un
                                    }
                    plot_id = 'prop-{}-{}-{}'.format('unique' if un else 'notunique',
                                                'max_duration_None' if (max_duration is None) else 'max_duration_' + str(max_duration),
                                                'stick_num_' + str(with_stick_num))

                    if with_stick_num:
                        rand_running = np.random.choice(list(set(astroA.indices_d[running_id]) - set(astroA.indices_d[stick_id])), size=len(astroA.indices_d[stick_id]), replace=False)
                        rand_no_running = np.random.choice(list(set(astroA.indices_d[rest_id]) - set(astroA.indices_d[stick_id])), size=len(astroA.indices_d[stick_id]), replace=False)
                    else:
                        rand_running = list(set(astroA.indices_d[running_id]) - set(astroA.indices_d[stick_id]))
                        rand_no_running = list(set(astroA.indices_d[rest_id]) - set(astroA.indices_d[stick_id]))

                    signal_delays_stick_np, peak_delays_stick_np = aqua_utils.get_delay_info_from_res(astroA.indices_d[stick_id], astroA.res_d, **delay_info_args)
                    signal_delays_running_np, peak_delays_running_np = aqua_utils.get_delay_info_from_res(rand_running, astroA.res_d, **delay_info_args)
                    signal_delays_no_running_np, peak_delays_no_running_np = aqua_utils.get_delay_info_from_res(rand_no_running, astroA.res_d, **delay_info_args)

                    stick_prop = np.zeros([max_delay-min_delay+1])
                    running_prop = np.zeros([max_delay-min_delay+1])
                    no_running_prop = np.zeros([max_delay-min_delay+1])

                    for i, delay_x in enumerate(range(min_delay, max_delay+1)):
                        stick_prop[i] = float(np.sum(signal_delays_stick_np == delay_x)) / len(signal_delays_stick_np)
                        running_prop[i] = float(np.sum(signal_delays_running_np == delay_x)) / len(signal_delays_running_np)
                        no_running_prop[i] = float(np.sum(signal_delays_no_running_np == delay_x)) / len(signal_delays_no_running_np)

                    x_l = [np.arange(min_delay, max_delay+1) for i in range(3)]
                    y_l = [stick_prop, running_prop, no_running_prop]

                    figs[plot_id] = plotly_utils.plot_scatter_mult(x_l, y_l, name_l=['stick', 'running', 'rest'], mode='lines', title='scatter', x_title='Delay (s)', y_title='Events')
                    plotly_utils.apply_fun_axis_fig(figs[plot_id], lambda x : x / astroA.fr, axis='x')
        return figs
    '''

    def get_triplet_plots(self, astroA, n_bins):
        fig_triplets = {}
        fig_radii_border = None
        for k in astroA.event_subsets.keys():
            print('THE KEY??', k)
            if astroA.aqua_bound == False:
                print('Plot triplet requires aqua bound to be true')
                return None, None
            #Find event centroids:
            #For each event in x2D extract 2D coordinates as mask and get event centroid coordinates
            #event_centroids = aqua_utils.get_event_centroids_from_x2D(astroA.res_d['x2D'], (astroA.input_shape[0], astroA.input_shape[1]))
            border_mask = astroA.res_d['border_mask']
            clandmark_center = astroA.res_d['clandmark_center']
            event_distances_from_center_micrometers = astroA.res_d['clandmark_distAvg'][astroA.event_subsets[k]]
            event_distances_from_center = event_distances_from_center_micrometers / astroA.spatial_res

            print('MAX EVENT DISTANCE:', np.max(event_distances_from_center))
            event_durations = astroA.res_d['tEnd'][astroA.event_subsets[k]] - astroA.res_d['tBegin'][astroA.event_subsets[k]]
            event_areas = astroA.res_d['area'][astroA.event_subsets[k]]

            n_events_arr_norm, n_events_i_arr, area_bins, r_bins = aqua_utils.radius_event_extraction(event_distances_from_center, clandmark_center, border_mask, n_bins=n_bins)
            event_distances_from_center_bins_l = []
            event_areas_bins_l = []
            event_durations_bins_l = []

            for event_inds in n_events_i_arr:
                event_distances_from_center_bins_l.append(event_distances_from_center[event_inds])
                event_areas_bins_l.append(event_areas[event_inds])
                event_durations_bins_l.append(event_durations[event_inds])

            print('areas', area_bins)
            print('total events norm', n_events_arr_norm)
            print('total events', [len(n) for n in n_events_i_arr])
            print('rbins:', r_bins)

            if 0 in [len(n) for n in n_events_i_arr]:
                print('not enough events for key: ', k)
                continue

            border_mask_temp = np.copy(astroA.res_d['border_mask'])
            #When indexing 2D array for x,y coordinates we need to index arr[row][col] = arr[y][x] so we flip the coordinates
            clandmark_center_flipped = (astroA.res_d['clandmark_center'][1], astroA.res_d['clandmark_center'][0])
            clandmark_center_flipped_int = (int(clandmark_center_flipped[0]), int(clandmark_center_flipped[1]))

            r_bin_diff = r_bins[1] - r_bins[0]
            #Radius bins of triplet plot on top of heatmap
            for i in range(border_mask_temp.shape[0]):
                for j in range(border_mask_temp.shape[1]):
                    v = border_mask_temp[i, j]
                    if v != 0:
                        r_dist = aqua_utils.get_euclidean_distances(clandmark_center_flipped, [i, j])
                        search_ind_r = np.searchsorted(r_bins, r_dist, side='right')
                        if search_ind_r == len(r_bins):
                            border_mask_temp[i, j] *= (r_bins[-1]+r_bin_diff)
                            #print('DISTANCE LARGER THAN MAX EVENT??', r_dist)
                        else:
                            border_mask_temp[i, j] *= r_bins[search_ind_r-1]
            #print('CLANDMARK CENTER', astroA.res_d['clandmark_center'])
            border_mask_temp[clandmark_center_flipped_int] -= r_bins[1]
            if fig_radii_border == None:
                fig_radii_border = plotly_utils.plot_contour(border_mask_temp, title='radius_extension_from_center', height=1000, width=1000,
                                                            color_bar_title='Radius (pixels)')
            print('N EVENTS HERE??', len(n_events_arr_norm[:-1]))
            fig_triplets[k] = plotly_utils.plot_event_triplet(num_events_bins=n_events_arr_norm[:-1],
                                                  distances_bins=r_bins[:-1],
                                                  sizes_bins_lists=event_areas_bins_l[:-1],
                                                  durations_bins_lists=event_durations_bins_l[:-1],
                                                  height=1000,
                                                  width=1000,
                                                  spatial_res=astroA.spatial_res,
                                                  fr=(1.0/astroA.fr_inv),
                                                  title=k + '_event_triplet_plot')
            break
        return fig_triplets, fig_radii_border

        '''
        #TODO save in file
        for k in event_subsets.keys():
            print('Number of events: {} - {} ind_size {}'.format(k , len(event_subsets[k]), len(indices_d[k])))

        print('Number of SHORT, MEDIUM AND LONG duration signals in different behavioural states')
        #How are the indices split between short, medium and long during running, stick, ...
        for k in indices_d.keys():
            short_signals_len = np.sum(all_durations_class_d[k] == 1)
            medium_signals_len = np.sum(all_durations_class_d[k] == 2)
            long_signals_len = np.sum(all_durations_class_d[k] == 3)
            total_signals = short_signals_len + medium_signals_len + long_signals_len
            print('{:20s}:\tshort\t{}\tmedium\t{}\tlong\t{}\tTotal signals\t{}\tlen\t({:7d})'.format(k, short_signals_len, medium_signals_len, long_signals_len, total_signals, len(indices_d[k])))
        '''

    def write_csv_duration_splits(self, astroA, path):
        #How are the indices split between short, medium and long during running, stick, ...
        with open(os.path.join(path), mode='w') as csv_file:
            writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['behaviour', 'short', 'medium', 'long', 'total events', 'total frames'])
            for k in astroA.indices_d.keys():
                short_signals_len = np.sum(astroA.all_durations_class_d[k] == 1)
                medium_signals_len = np.sum(astroA.all_durations_class_d[k] == 2)
                long_signals_len = np.sum(astroA.all_durations_class_d[k] == 3)
                total_signals = short_signals_len + medium_signals_len + long_signals_len

                short_signals_ratio = general_utils.truncate(short_signals_len/total_signals, 2)
                long_signals_ratio = general_utils.truncate(long_signals_len/total_signals, 2)
                medium_signals_ratio = general_utils.truncate(medium_signals_len/total_signals, 2)
                writer.writerow([k, short_signals_ratio, medium_signals_ratio, long_signals_ratio, total_signals, len(astroA.indices_d[k])])

    def save_xcorr_pairs_align_results_csv(self, save_path, astro_pair_l, pair_fakes_before, pair_fakes_after, pair_corrs_before, pair_corrs_after):
        with open(os.path.join(save_path), mode='w') as csv_file:
            writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            for i, astro_pair in enumerate(astro_pair_l):
                name = astro_pair[0].print_id + '-' + astro_pair[1].print_id

                writer.writerow(['Name'])
                writer.writerow([name])
                writer.writerow(['Samples before'])
                writer.writerow([general_utils.truncate(v, 4) for v in pair_fakes_before[i]])
                writer.writerow(['Samples after'])
                writer.writerow([general_utils.truncate(v, 4) for v in pair_fakes_after[i]])
                writer.writerow(['Corr before'])
                writer.writerow([general_utils.truncate(pair_corrs_before[i], 4)])
                writer.writerow(['Corr after'])
                writer.writerow([general_utils.truncate(pair_corrs_after[i], 4)])
                writer.writerow([])
                writer.writerow([])

    def get_duration_split_differences_from_default(self, astroA):
        #How are the indices split between short, medium and long during running, stick, ...
        relative_ratios = {}
        ratios = {}
        lengths = {}
        ks = np.array(self.filter_keys(astroA))

        for k in ks:
            relative_ratios[k] = {}
            ratios[k] = {}
            lengths[k] = {}

            short_signals_len = np.sum(astroA.all_durations_class_d[k] == 1)
            medium_signals_len = np.sum(astroA.all_durations_class_d[k] == 2)
            long_signals_len = np.sum(astroA.all_durations_class_d[k] == 3)
            total_signals = short_signals_len + medium_signals_len + long_signals_len

            lengths[k]['short'] = short_signals_len
            lengths[k]['medium'] = medium_signals_len
            lengths[k]['long'] = long_signals_len

            short_signals_ratio = general_utils.truncate(short_signals_len/total_signals, 4)
            long_signals_ratio = general_utils.truncate(long_signals_len/total_signals, 4)
            medium_signals_ratio = general_utils.truncate(medium_signals_len/total_signals, 4)

            ratios[k]['short'] = short_signals_ratio
            ratios[k]['medium'] = medium_signals_ratio
            ratios[k]['long'] = long_signals_ratio

            for dk in ratios[k].keys():
                relative_ratios[k][dk] = ratios[k][dk] - ratios['default'][dk] + 0.00001

        x = ks
        y_l = [[relative_ratios[k][duration_k] for k in ks] for duration_k in ['short', 'medium', 'long']]
        text_values_l = [[lengths[k][duration_k] for k in ks] for duration_k in ['short', 'medium', 'long']]
        legends=['short', 'medium', 'long']
        fig = plotly_utils.plot_group_bar(x=x, y_l=y_l, text_values_l=text_values_l, legends=legends, x_title='Behaviour', y_title='Relative difference to default', title='Relative difference in short,medium,long signals to default')
        return fig

    def get_signal_fig(self, astroA, event_i):
        x1 = np.arange(len(astroA.res_d['dff_only'][event_i]))
        y1 = astroA.res_d['dff_only'][event_i]
        divisor = astroA.res_d['dff_only'].shape[1]
        t_begin = int(astroA.res_d['tBegin'][event_i] % divisor)
        t_end = int(astroA.res_d['tEnd'][event_i] % divisor)
        if t_begin > t_end:
            print('Tbegin > tEnd')
            return None

        fig = plotly_utils.plot_scatter_signal(x=x1, y=y1, begin_i=t_begin, end_i=t_end, mode='lines', title='Signal', x_title='', y_title='')
        return fig

    def get_signal_figs_samples(self, astroA, sample_num=20):
        event_sample_inds = np.random.choice(len(astroA.res_d['tBegin']), sample_num, replace=False)
        figs = []
        for event_i in event_sample_inds:
            figs.append(self.get_signal_fig(astroA, event_i))
        return figs

    def get_signal_bk_figs_samples(self, astroA, sample_num=10):
        figs = {}
        for bk in astroA.event_subsets.keys():
            sample_num_x = min(len(astroA.event_subsets[bk]), sample_num)
            event_sample_inds = np.random.choice(astroA.event_subsets[bk], sample_num_x, replace=False)
            figs[bk] = []

            for event_i in event_sample_inds:
                figs[bk].append(self.get_signal_fig(astroA, event_i))
        return figs

######Other analysis#######
    '''
    def get_frame_split_max_corrs_plots(self, astroA):
        #fake_corr_d, splits_corr_d = self.read_corr_data(astroA)
        fake_corr_d = self.read_corr_data(astroA)
        figs = {}
        for prob_k in splits_corr_d.keys():
            if prob_k not in fake_corr_d.keys():
                continue

            max_corrs = {}
            max_corrs['fake'] = fake_corr_d[prob_k]['max_corr_l']
            for frame_split in splits_corr_d[prob_k].keys():
                if len(splits_corr_d[prob_k][frame_split]['max_corr_l']) != 0:
                    max_corrs[frame_split + ' frames'] = splits_corr_d[prob_k][frame_split]['max_corr_l']

            x = list(max_corrs.keys())
            y = [max_corrs[k] for k in x]
            figs[prob_k] = plotly_utils.plot_point_box_revised(x, y, title='Max correlations on different frame splits', x_title='Frames ({} fps)'.format(general_utils.truncate(astroA.fr, 2)), y_title='Max correlation value')
        return figs
    '''
    def get_compare_max_corrs_plots(self, astro_l_pair):
        corr_pair_d = self.read_corr_pair_data(astro_l_pair)
        figs = {}
        for prob_k in corr_pair_d.keys():
            print(prob_k)
            self_max_corr = corr_pair_d[prob_k]['self']['max_corr']
            compare_max_corr = corr_pair_d[prob_k]['compare']['max_corr']
            samples_max_corr_l = corr_pair_d[prob_k]['samples']['max_corr_l']

            x = ['self: day {}'.format(astro_l_pair[0].day),
                 'comparison: day {}'.format(astro_l_pair[1].day),
                 'samples']
            y = [[self_max_corr], [compare_max_corr], samples_max_corr_l]

            figs[prob_k] = plotly_utils.plot_point_box_revised(x, y, title='Max correlations', x_title='', y_title='Max correlation value')
        return figs

    def get_compare_align_plots(self, astro_l_pair):
        figs = {}
        align_setting_l = ['xcorr', 'clandmark']
        move_vector_d = {'xcorr' : None, 'clandmark' : None}

        for align_setting in align_setting_l:
            d_temp = compare_astro_utils.alignment_counter(astro_l_pair[0], astro_l_pair[1],
                                                            n_fake_samples=0,
                                                            align_setting=align_setting,
                                                            eval_setting='counter',
                                                            fake_sample_setting='from_grid',
                                                            behaviour='default',
                                                            p=0.05)
            move_vector_d[align_setting] = d_temp['move_vector']

            figs[align_setting] = {}
            for pk in self.filter_probs:
                d = compare_astro_utils.alignment_counter(astro_l_pair[0], astro_l_pair[1],
                                                            n_fake_samples=self.n_samples_corr_fake,
                                                            align_setting='param',
                                                            eval_setting='counter',
                                                            fake_sample_setting='from_grid',
                                                            move_vector=move_vector_d[align_setting],
                                                            p=pk)

                x = ['fake_samples', 'day {} aligned'.format(astro_l_pair[1].day), 'day {} self aligned'.format(astro_l_pair[0].day)]
                y = [d['num_fake'], [d['num_compare']], [d['num_self']]]

                figs[align_setting][pk] = plotly_utils.plot_point_box_revised(x, y, title='Day {} with day {} alignment. Comparison of intersection size to random samples - p = {}'.format(astro_l_pair[0].day, astro_l_pair[1].day, pk),
                                                                            x_title='', y_title='Aligned intersection size')
        return figs

    def get_compare_align_plots_xcorr(self, astro_l_pair, align_setting='xcorr', dff_mode=False, behaviour='default'):
        d_temp = compare_astro_utils.alignment_counter(astro_l_pair[0], astro_l_pair[1],
                                                        n_fake_samples=0,
                                                        align_setting=align_setting,
                                                        eval_setting='xcorr',
                                                        fake_sample_setting='from_astro',
                                                        p=1,
                                                        behaviour='default') #Here I use default to homogenize move_vector result over diffeerent behaviours
        move_vector = d_temp['move_vector']
        d = compare_astro_utils.alignment_counter(astro_l_pair[0], astro_l_pair[1],
                                                    n_fake_samples=self.n_samples_corr_fake,
                                                    align_setting='param',
                                                    eval_setting='xcorr',
                                                    fake_sample_setting='from_astro',
                                                    move_vector=move_vector,
                                                    p=1,
                                                    behaviour=behaviour)

        x = ['fake_samples', 'day {} aligned'.format(astro_l_pair[1].day)]
        y = [d['num_fake'], [d['num_compare']]]

        tstat, pvalue = ttest_ind_from_stats(np.mean(y[0]), np.std(y[0]), len(y[0]), np.mean(y[1]), np.std(y[1]), len(y[1]))
        print('NUM COMPARE: {}, mode {} behaviour {}'.format(d['num_compare'], dff_mode, behaviour))
        fig = plotly_utils.plot_point_box_revised(x, y, title='{} Day {} with day {} alignment. Comparison to random. p={:.2e}'.format(behaviour, astro_l_pair[0].day, astro_l_pair[1].day, pvalue),
                                                                    x_title='', y_title='Aligned xcorr value')
        return fig

    def get_day_heatmaps_scaled(self, astroA_pair, bh='default', dff_mode=False):
        if dff_mode==True:
            raise NotImplementedError()

        if (bh not in astroA_pair[0].event_grids_1min) or (bh not in astroA_pair[0].event_grids_1min):
            return None

        day_0_grid = astroA_pair[0].event_grids_1min[bh] if dff_mode else astroA_pair[0].event_grids_1min_dff[bh]
        day_1_grid = astroA_pair[1].event_grids_1min[bh] if dff_mode else astroA_pair[1].event_grids_1min_dff[bh]

        max_0 = np.max(day_0_grid)
        max_1 = np.max(day_1_grid)

        if max_1 > max_0:
            contour_day_0, details_d = plotly_utils.plot_contour_threshold(day_0_grid, threshold_perc=1.0, title=bh + '_event grid', with_details=True)
            min_v, max_v = details_d['min'], details_d['max']
            contour_day_x = plotly_utils.plot_contour_threshold(day_1_grid, threshold_perc=None, set_min_v=min_v, set_max_v=max_v, title=bh+ '_event_grid')
        else:
            contour_day_x, details_d = plotly_utils.plot_contour_threshold(day_1_grid, threshold_perc=1.0, title=bh + '_event grid', with_details=True)
            min_v, max_v = details_d['min'], details_d['max']
            contour_day_0 = plotly_utils.plot_contour_threshold(day_0_grid, threshold_perc=None, set_min_v=min_v, set_max_v=max_v, title=bh+ '_event_grid')
        return {'contour_0' : contour_day_0, 'contour_x' : contour_day_x}

    def get_individual_heatmaps_threshold_scaled(self, astroA, bh='default', threshold=0.7, num_samples=3, dff_mode=False, with_arr=False):
        if dff_mode==True:
            raise NotImplementedError()

        arrs_d = {}


        if (bh not in astroA.event_grids_1min) or (bh not in astroA.event_grids_1min):
            return None
        contour, details_d = plotly_utils.plot_contour_threshold(astroA.event_grids_1min[bh] if not dff_mode else astroA.event_grids_1min_dff[bh], title=bh + '_event grid', threshold_perc=threshold, with_details=True)
        min_v, max_v = details_d['min'], details_d['max']
        f, sample_l = self.get_random_astrocyte_plot(astroA, bh=bh, with_samples=True, num_samples=num_samples)
        contour_random_l = []

        if with_arr:
            arrs_d['arr'] = details_d['arr']
            arrs_d['arr_r'] = []
        for sample in sample_l:
            contour_random, details_r_d = plotly_utils.plot_contour_threshold(sample, threshold_perc=None, set_min_v=min_v, set_max_v=max_v, title=bh + '_random_event_grid', with_details=True)
            contour_random_l.append(contour_random)
            arrs_d['arr_r'].append(details_r_d['arr'])

        if with_arr:
            return {'contour' : contour,
                    'contour_random' : contour_random_l,
                    'arrs_d' : arrs_d}

        return {'contour' : contour,
                'contour_random' : contour_random_l}


    def get_compare_frame_split_plots(self, astroA):
        figs = {}
        data_d = {}

        event_grid_splits_d = {}
        for split_frames in self.num_frames_splits_l:
            event_grid_splits_d[split_frames] = compare_astro_utils.split_astro_grid(astroA, split_frames=split_frames, bk='default')

        for pk in self.filter_probs:
            figs[pk] = {}
            data_d[pk] = {}
            for split_frames in self.num_frames_splits_l:
                event_grid_splits_l = event_grid_splits_d[split_frames]
                pairs = [(i, j ) for i in range(len(event_grid_splits_l)) for j in range(i+1, len(event_grid_splits_l))]
                data_d[pk][split_frames] = {'num_fake_l' : [], 'num_compare_l' : [], 'num_self_l' : [], 'num_fake_ratio_l' : [], 'num_compare_ratio_l' : []}

                if len(pairs) > self.max_split_comparison_samples :
                    print('Max comparisons > len pairs, {} > {}'.format(self.max_split_comparison_samples, len(pairs)))
                    pairs_perm = np.random.permutation(pairs)
                    pairs = pairs_perm[:self.max_split_comparison_samples]

                for (i, j) in pairs:
                    #Pretty much takes p highest, and calculates intersections with self, compare and fake samples
                    d = compare_astro_utils.alignment_counter(astroA, astroA,
                                      grid_target=event_grid_splits_l[i], grid_source=event_grid_splits_l[j],
                                      n_fake_samples=1, align_setting='param',
                                      move_vector=[0,0], p=pk)

                    data_d[pk][split_frames]['num_fake_l'].append(d['num_fake'][0])
                    data_d[pk][split_frames]['num_compare_l'].append(d['num_compare'])
                    data_d[pk][split_frames]['num_self_l'].append(d['num_self'])
                    data_d[pk][split_frames]['num_fake_ratio_l'].append(d['num_fake'][0]/float(d['num_self']))
                    data_d[pk][split_frames]['num_compare_ratio_l'].append(d['num_compare']/float(d['num_self']))

                    #print('num self:', d['num_self'])
                    #print('num fake:', d['num_fake'][0])
                    #print('num compare:', d['num_compare'])
                    #print('ratio fake:', data_d[pk][split_frames]['num_fake_ratio_l'])
                    #print('ratio compare:', data_d[pk][split_frames]['num_compare_ratio_l'])
            x = []
            y = [[], []]
            for split_frames in self.num_frames_splits_l:
                if len(data_d[pk][split_frames]['num_fake_ratio_l']) == 0:
                    continue
                f_m = np.mean(data_d[pk][split_frames]['num_fake_ratio_l'])
                f_s = np.std(data_d[pk][split_frames]['num_fake_ratio_l'])
                f_l = len(data_d[pk][split_frames]['num_fake_ratio_l'])

                c_m = np.mean(data_d[pk][split_frames]['num_compare_ratio_l'])
                c_s = np.std(data_d[pk][split_frames]['num_compare_ratio_l'])
                c_l = len(data_d[pk][split_frames]['num_compare_ratio_l'])
                tstat, pvalue = ttest_ind_from_stats(c_m, c_s, c_l, f_m, f_s, f_l)

                for i in range(len(data_d[pk][split_frames]['num_fake_ratio_l'])):
                    x.append('~ {} minutes <br /> p = {:.1e}'.format(general_utils.truncate(split_frames/(astroA.fr*60), 1), pvalue))

                y[0].extend(data_d[pk][split_frames]['num_fake_ratio_l'])
                y[1].extend(data_d[pk][split_frames]['num_compare_ratio_l'])
                #print('TSTAT: {} PVALUE: {}'.format(tstat, pvalue))

            figs[pk] = plotly_utils.plot_multi_point_box(x, y, names=['Fake', 'Compare'], title='Splits comparison - Top {}%: '.format(pk*100),
                                                                        x_title='', y_title='Intersection ratio')
        return figs

    def get_compare_corrs_samples_plots(self, astroA_l):
        figs = {}
        corr_pair_d = self.read_corr_pair_data(astroA_l)
        for pk in corr_pair_d.keys():
            astro_filt_l, astro_all_filt, astro_nz_bool_l, astro_all_nz_bool = compare_astro_utils.get_filters_compare(astroA_l, p=pk)

            ids = ['astro_a', 'astro_b', 'border', 'sample_1', 'sample_2', 'sample_3']

            print('COOR PAIR D', corr_pair_d)
            print(pk)
            print(len(corr_pair_d[pk]['samples']['sample_l']))

            grid_l = [astro_filt_l[0],
                      astro_filt_l[1],
                      astro_nz_bool_l[1].astype(int),
                      corr_pair_d[pk]['samples']['sample_l'][0],
                      corr_pair_d[pk]['samples']['sample_l'][1],
                      corr_pair_d[pk]['samples']['sample_l'][2]]

            titles = ['Astrocyte A day {}'.format(astroA_l[0].day),
                      'Astrocyte B day {}'.format(astroA_l[1].day),
                      'Border Astro B',
                      'Sample 1',
                      'Sample 2',
                      'Sample 3']

            for i, grid in enumerate(grid_l):
                if i == 0:
                    figs[pk] = {}

                figs[pk][ids[i]] = plotly_utils.plot_contour(grid, title=titles[i])
        return figs

    def get_frame_split_example_plots(self, astroA):
        figs = {}
        for pk in self.filter_probs:
            figs[pk] = {}
            for num_frames_split in self.num_frames_splits_l:
                event_grid_splits_l = compare_astro_utils.split_astro_grid(astroA, split_frames=num_frames_split, bk='default')
                if len(event_grid_splits_l) < 2:
                    continue
                inds = np.random.choice(len(event_grid_splits_l), 2, replace=False)
                grid_l_pair = [event_grid_splits_l[inds[0]], event_grid_splits_l[inds[1]]]
                astro_filt_l_tmp, astro_all_filt_tmp, astro_nz_bool_l_tmp, astro_all_nz_bool_tmp = compare_astro_utils.get_filters_compare_from_grids(grid_l_pair, p=float(pk))
                figs[pk][str(num_frames_split)] = plotly_utils.plot_contour_multiple(astro_filt_l_tmp, title='Frames: {} top {}%'.format(num_frames_split, pk))
        return figs

    def get_compare_full_self_frame_split_plot_xcorr(self, astroA, minute_frame_splits_l=None):
        '''
        Grid 1 is normalized event heatmap of astroA
        We apply frame splits of 0.5, 1, 2, 5, 10 ,15, 20, 25, 30 minutes splits to obtain grid 2.
        Then we get cross correlation between the 2
        '''
        xcorr_split_corrs_d = {}

        grid_1 = astroA.event_grids['default']

        if minute_frame_splits_l is None:
            minute_frame_splits_l = self.num_frames_splits_m_l[0:3][::-1]
        #frame_splits_l_temp = [int(np.round(astroA.fr*split_frames_m*60)) for split_frames_m in self.num_frames_splits_m_l[::-1]]
        frame_splits_l_temp = [int(np.round(astroA.fr*split_frames_m*60)) for split_frames_m in minute_frame_splits_l]
        frame_splits_l_temp.insert(0, len(astroA.indices_d['default']))

        for split_frames in frame_splits_l_temp:
            xcorr_split_corrs_d[split_frames] = []

            print('Split frames (self): {}'.format(split_frames))

            event_grid_splits_l = compare_astro_utils.split_astro_grid(astroA, split_frames=split_frames, bk='default')

            for i in range(len(event_grid_splits_l)):
                grid_2 = event_grid_splits_l[i]
                res = compare_astro_utils.alignment_counter(astroA, astroA, grid_target=grid_1, grid_source=grid_2, n_fake_samples=0,
                                            align_setting='param', eval_setting='xcorr', fake_sample_setting='from_grid',
                                            move_vector=[0,0], p=1, dff_mode=False, behaviour='default', filter_duration=(None, None),
                                            with_output_details=False, border_nan=True)

                corr_res = res['num_compare']
                xcorr_split_corrs_d[split_frames].append(corr_res)

        x = ['~ {}'.format(np.round(general_utils.truncate(split_frames/(astroA.fr*60), 1), decimals=1)) for split_frames in frame_splits_l_temp]
        y = [xcorr_split_corrs_d[split_frames] for split_frames in frame_splits_l_temp]

        x_fixed = []
        y_fixed = []

        for i in range(len(y)):
            if len(y[i]) == 0:
                continue
            x_fixed.append(x[i])
            y_fixed.append(y[i])

        fig_a = plotly_utils.plot_point_box_revised(x_fixed, y_fixed, title='Split correlation to full', x_title='Minutes splits', y_title='Correlation (max = 1)')
        fig_b = plotly_utils.plot_scatter_error(x_fixed, y_fixed, title='Split correlation to full', x_title='Minutes splits', y_title='Correlation (max = 1)')

        return fig_a, fig_b

    def get_compare_full_self_results_alt(self, astroA, cut_duration_min=70, minute_frame_splits_l=None, save_pkl_path=None):
        '''
        Grid 1 is normalized event heatmap of astroA
        We apply frame splits of 0.5, 1, 2, 5, 10 ,15, 20, 25, 30 minutes splits to obtain grid 2.
        Then we get cross correlation between the 2
        '''
        res_d = {}
        if (cut_duration_min is not None):
            if (astroA.total_minutes < cut_duration_min):
                return None

        if os.path.isfile(save_pkl_path):
            print('FILE EXISTS')
            res_d = saving_utils.load_pickle(save_pkl_path)
        else:
            xcorr_split_corrs_d = {}

            #How many frames are cut_duration_min minutes? We cut up to this point
            num_frames_cut_duration = int(cut_duration_min * astroA.minute_frames)

            default_ind = astroA.indices_d['default']
            cut_ind = astroA.indices_d['default'][:num_frames_cut_duration]

            print(default_ind)
            print(cut_ind)
            event_subsets_temp = aqua_utils.get_event_subsets({'default' : default_ind, 'cut' : cut_ind}, astroA.res_d, after_i=0, before_i=0, to_print=False)
            cut_event_subsets = event_subsets_temp['cut']

            print('MAX BEGIN TIME', np.max(astroA.res_d['tBegin'][cut_event_subsets]))
            print('MAX IND', len(astroA.indices_d['default']))
            print('NUM FRAMES CUT', num_frames_cut_duration)

            grid_1 = aqua_utils.get_event_grid_from_x2D(astroA.res_d['x2D'][cut_event_subsets], (astroA.input_shape[0], astroA.input_shape[1]))

            if minute_frame_splits_l is None:
                minute_frame_splits_l = self.num_frames_splits_m_l[0:3][::-1]

            frame_splits_l_temp = [int(np.round(astroA.fr*split_frames_m*60)) for split_frames_m in minute_frame_splits_l]
            frame_splits_l_temp.insert(0, len(cut_ind))

            for split_frames in frame_splits_l_temp:
                xcorr_split_corrs_d[split_frames] = []

                print('Split frames (self): {}'.format(split_frames))

                event_grid_splits_l = compare_astro_utils.split_astro_grid(astroA, split_frames=split_frames, bk='default', inds_subset=cut_ind)

                for i in range(len(event_grid_splits_l)):
                    grid_2 = event_grid_splits_l[i]
                    res = compare_astro_utils.alignment_counter(astroA, astroA, grid_target=grid_1, grid_source=grid_2, n_fake_samples=0,
                                                align_setting='param', eval_setting='xcorr', fake_sample_setting='from_grid',
                                                move_vector=[0,0], p=1, dff_mode=False, behaviour='default', filter_duration=(None, None),
                                                with_output_details=False, border_nan=True)

                    corr_res = res['num_compare']
                    xcorr_split_corrs_d[split_frames].append(corr_res)


            #x = minute_frame_splits_l
            x = ['~ {}'.format(np.round(general_utils.truncate(split_frames/(astroA.fr*60), 1), decimals=1)) for split_frames in frame_splits_l_temp]
            y = [xcorr_split_corrs_d[split_frames] for split_frames in frame_splits_l_temp]
            x_fixed = []
            y_fixed = []

            for i in range(len(y)):
                if len(y[i]) == 0:
                    continue
                x_fixed.append(x[i])
                y_fixed.append(y[i])

            res_d['x'] = x_fixed
            res_d['y'] = y_fixed
        if save_pkl_path is not None:
            saving_utils.save_pickle(res_d, save_pkl_path)

        return res_d



    def get_compare_full_self_frame_split_split_plot_xcorr(self, astroA):
        '''
        Grid 1 is normalized event heatmap of astroA
        We apply frame splits of minutes splits to obtain grid 2.
        Then we get cross correlation between the 2
        '''
        xcorr_split_corrs_d = {}
        grid_1 = astroA.event_grids['default']
        frame_splits_l_temp = [int(np.round(astroA.fr*split_frames_m*60)) for split_frames_m in self.num_frames_splits_splits_m_l[::-1]]

        print('????', frame_splits_l_temp, astroA.id)
        print(len(frame_splits_l_temp))
        for split_frames in frame_splits_l_temp:
            xcorr_split_corrs_d[split_frames] = []
            print('Split frames (self): {}'.format(split_frames))
            event_grid_splits_l = compare_astro_utils.split_astro_grid(astroA, split_frames=split_frames, bk='default')
            print('LEN EVENT GRIPD SPLITS L', len(event_grid_splits_l))
            if len(event_grid_splits_l) > 2:
                for i in range(len(event_grid_splits_l)):
                    grid_1 = event_grid_splits_l[i]
                    for j in range(i+1, len(event_grid_splits_l)):
                        grid_2 = event_grid_splits_l[j]
                        res = compare_astro_utils.alignment_counter(astroA, astroA, grid_target=grid_1, grid_source=grid_2, n_fake_samples=0,
                                                    align_setting='param', eval_setting='xcorr', fake_sample_setting='from_grid',
                                                    move_vector=[0,0], p=1, dff_mode=False, behaviour='default', filter_duration=(None, None),
                                                    with_output_details=False, border_nan=True)

                        corr_res = res['num_compare']
                        xcorr_split_corrs_d[split_frames].append(corr_res)
        x = ['~ {}'.format(np.round(general_utils.truncate(split_frames/(astroA.fr*60), 1), decimals=1)) for split_frames in frame_splits_l_temp]
        y = [xcorr_split_corrs_d[split_frames] for split_frames in frame_splits_l_temp]

        x_fixed = []
        y_fixed = []

        for i in range(len(y)):
            if len(y[i]) == 0:
                continue
            x_fixed.append(x[i])
            y_fixed.append(y[i])

        fig_a = plotly_utils.plot_point_box_revised(x_fixed, y_fixed, title='Split-split correlation', x_title='Minutes splits', y_title='Correlation (max = 1)')
        fig_b = plotly_utils.plot_scatter_error(x_fixed, y_fixed, title='Split-split correlation to full', x_title='Minutes splits', y_title='Correlation (max = 1)')
        return fig_a, fig_b


    def get_random_astrocyte_plot(self, astroA, num_samples=3, bh='default', with_samples=False):
        event_areas = astroA.res_d['area'][astroA.event_subsets[bh]] / (astroA.spatial_res**2)
        fig_l = []
        sample_l = []
        for i in range(num_samples):
            sample = compare_astro_utils.get_fake_astrocyte_sample_from_areas(astroA, event_areas, mode='append', filter_ratio=1)
            sample = (sample / len(astroA.indices_d[bh])) * astroA.minute_frames
            fig_l.append(plotly_utils.plot_contour(sample, title='Random event contour plot'))
            sample_l.append(sample)
        if with_samples:
            return fig_l, sample_l
        return fig_l

    def filter_keys(self, astroA):
        to_remove_k = list(set(self.behaviours_list_a) - set(list(astroA.event_subsets.keys())))
        print('keys to remove:  ', to_remove_k)
        print('new filter keys:', [k for k in self.behaviours_list_a if k not in to_remove_k])
        return [k for k in self.behaviours_list_a if k not in to_remove_k]

    def get_plot_first_last_x_min_behaviour(self, astroA, num_min=20, behaviour_ind='rest'):
        """
        Get plot of first and last twenty minutes of rest
        """
        num_min_frames = int(np.round(astroA.fr * num_min * 60))

        if len(astroA.indices_d[behaviour_ind]) < 2 * num_min_frames:
            return None

        first_inds = astroA.indices_d[behaviour_ind][:num_min_frames]
        last_inds = astroA.indices_d[behaviour_ind][-num_min_frames:]

        indices_d_temp = {'default' : astroA.indices_d['default'], 'first' : first_inds, 'last' : last_inds}
        event_subsets_temp = aqua_utils.get_event_subsets(indices_d_temp, astroA.res_d)

        event_grid_first = aqua_utils.get_event_grid_from_x2D(astroA.res_d['x2D'][event_subsets_temp['first']], (astroA.input_shape[0], astroA.input_shape[1]))
        event_grid_last = aqua_utils.get_event_grid_from_x2D(astroA.res_d['x2D'][event_subsets_temp['last']], (astroA.input_shape[0], astroA.input_shape[1]))

        return plotly_utils.plot_contour_multiple([event_grid_first, event_grid_last], title='{}'.format(behaviour_ind), subplot_titles=['First {} min'.format(num_min), 'Last {} min'.format(num_min)])

    def get_plot_x_min_rest_relative(self, astroA, num_min=20, behaviour_ind='rest'):
        """
        Compare correlation of first 20 minutes with subsequent 20 minutes. This is to see how the correlation degrades over time.
        """
        num_min_frames = int(np.round(astroA.fr * num_min * 60))
        if len(astroA.indices_d[behaviour_ind]) < 2 * num_min_frames:
            return None

        ind_split_l = []
        indices_d_temp = {}
        indices_d_temp['default'] = astroA.indices_d['default']

        #Get frame indices corresponding to each x min split
        for i in range(len(astroA.indices_d[behaviour_ind]) // num_min_frames):
            ind_split = astroA.indices_d[behaviour_ind][i*num_min_frames:(i+1)*num_min_frames]
            ind_split_l.append(ind_split)

            indices_d_temp[i] =  ind_split

        #Get event indices corresponting to each x min split
        event_subsets_temp = aqua_utils.get_event_subsets(indices_d_temp, astroA.res_d)

        #Get event grid corresponding to each x min split
        event_grid_split_l = []
        for i in range(len(astroA.indices_d[behaviour_ind]) // num_min_frames):
            event_grid_x = aqua_utils.get_event_grid_from_x2D(astroA.res_d['x2D'][event_subsets_temp[i]], (astroA.input_shape[0], astroA.input_shape[1]))
            event_grid_split_l.append(event_grid_x)

        corr_res_l = []
        #Calculate xcorr between first grid and rest
        for i in range(1, len(event_grid_split_l)):
            corr_res, _, move_vector, _ = correlation_utils.get_cross_correlation_2D_info_compare(event_grid_split_l[0], event_grid_split_l[i], normalize=True, mode='valid')
            corr_res_l.append(corr_res[0][0])

        corr_res_l = np.array(corr_res_l)

        return plotly_utils.plot_scatter([i for i in range(len(corr_res_l))], corr_res_l , mode='lines', title='scatter', x_title='', y_title='')

    def get_plot_compare_behaviour_correlation(self, astro_l_pair, dff_mode=False):
        behaviour_l = ['rest', 'running', 'stick', 'whisker']
        results = {}
        #run, rest, stick, whisker
        #Get correlations bh-bh day 0, day 1

        #Step 1: obtain move vector for alignment between day 0 and day x
        d_temp = compare_astro_utils.alignment_counter(astro_l_pair[0], astro_l_pair[1],
                                                        n_fake_samples=0,
                                                        align_setting='xcorr',
                                                        eval_setting='xcorr',
                                                        fake_sample_setting='from_astro',
                                                        p=1,
                                                        behaviour='default',
                                                        dff_mode=dff_mode)
        move_vector = d_temp['move_vector']

        #Step 2: obtain correlation
        for bh_i in behaviour_l:
            d = compare_astro_utils.alignment_counter(astro_l_pair[0], astro_l_pair[1],
                                                        n_fake_samples=0,
                                                        align_setting='param',
                                                        eval_setting='xcorr',
                                                        fake_sample_setting='from_astro',
                                                        move_vector=move_vector,
                                                        p=1,
                                                        behaviour=bh_i,
                                                        dff_mode=dff_mode)
            results[bh_i + '_' + bh_i] = d['num_compare']

        #Get correlations bh_i-bh_j same day
        for bh_i in range(len(behaviour_l)):
            for bh_j in range(bh_i+1, len(behaviour_l)):
                if bh_i == bh_j:
                    continue
                for i in range(2):
                    d = compare_astro_utils.alignment_counter(astro_l_pair[i], astro_l_pair[i],
                                                                n_fake_samples=0,
                                                                align_setting='param',
                                                                eval_setting='xcorr',
                                                                fake_sample_setting='from_astro',
                                                                move_vector=[0,0],
                                                                p=1,
                                                                behaviour=[behaviour_l[bh_i], behaviour_l[bh_j]],
                                                                dff_mode=dff_mode)

                    results[behaviour_l[bh_i] + '_' + behaviour_l[bh_j] + '_' + str(i)] = d['num_compare']

        ###
        x = list(results.keys())
        y = [[results[x_k]] for x_k in x]

        return plotly_utils.plot_point_box_revised(x, y, margin_b=400)


    def measure_distribution_plot(self, astroA_l, bh, measure, num_bins=10, min_measure=0, max_measure=0, measure_name=''):
        '''
        Default min is 0
        '''
        measure_d = {}
        for astroA in astroA_l:
            if bh in astroA.event_subsets:
                measure_d[astroA.print_id] = astroA.res_d[measure][astroA.event_subsets[bh]]

        measure_counts_d = {}

        for k in measure_d.keys():
            if min_measure is not None:
                measure_d[k] = measure_d[k][measure_d[k] >= min_measure]
                min_range = min_measure
            if max_measure is not None:
                measure_d[k] = measure_d[k][measure_d[k] <= max_measure]
                max_range = max_measure

            if min_measure is None:
                min_range = np.min([np.min(measure_d[k]) for k in measure_d.keys()])
            if max_measure is None:
                max_range = np.max([np.max(measure_d[k]) for k in measure_d.keys()])

            measure_counts_d[k], bins_arr = np.histogram(measure_d[k], bins=num_bins, range=(min_range, max_range))
            measure_counts_d[k] = measure_counts_d[k] / np.sum(measure_counts_d[k])

        y_l = [[measure_counts_d[k][i] for k in measure_counts_d.keys()] for i in range(num_bins)]
        x = bins_arr

        x_title = measure_name
        if measure_name == 'duration':
            x_title += ' (s)'
        if measure_name == 'size':
            #TODO
            x_title += ''
        fig = plotly_utils.plot_scatter_error(x, y_l, mode='lines', title='{}-{} distribution'.format(bh, measure_name), x_title=measure_name, y_title='')

        return fig, x, y_l

    def measure_distribution_bh_compare_plot(self, astroA_l, bh_l, measure, num_bins=10, min_measure=0, max_measure=0, measure_name='', confidence=True, with_stats=True):
        bh_y_d = {}
        x_l = []
        for bh in bh_l:
            _, x, y_l = self.measure_distribution_plot(astroA_l, bh, measure=measure, num_bins=num_bins, min_measure=min_measure, max_measure=max_measure, measure_name=measure_name)
            x_l.append(x)
            if confidence:
                bh_y_d[bh] = y_l
            else:
                bh_y_d[bh] = [np.mean(y) for y in y_l]

        bh_k_l = list(bh_y_d.keys())
        bh_y_l = [bh_y_d[bh] for bh in bh_k_l]

        x_title = measure_name
        if measure_name == 'duration':
            x_title += ' (s)'

        return plotly_utils.plot_scatter_mult(x_l=x_l, y_l_l=bh_y_l,
                                                name_l=bh_k_l, title='{}s distribution'.format(measure_name),
                                                x_title=x_title, y_title='',
                                                xrange=(min_measure, max_measure), confidence=confidence, with_stats=True)

    def measure_distribution_bh_compare_plot_exponential_fit(self, astroA_l, bh_l, measure, num_bins=10, min_measure=0, max_measure=0, measure_name='', confidence=True, with_stats=True, with_log=True):
        bh_y_d = {}
        x_l = []
        for bh in bh_l:
            print(bh)
            print(measure)
            _, x, y_l = self.measure_distribution_plot(astroA_l, bh, measure=measure, num_bins=num_bins, min_measure=min_measure, max_measure=max_measure, measure_name=measure_name)
            x_l.append(x[:-1])
            if confidence:
                bh_y_d[bh] = y_l
            else:
                bh_y_d[bh] = [np.mean(y) for y in y_l]

        if with_log:
            if min_measure is not None:
                min_measure=np.log(min_measure)
            if max_measure is not None:
                max_measure=np.log(max_measure)
            x_l = list(np.log(np.array(x_l)))
        bh_k_l = list(bh_y_d.keys())
        bh_y_l = [bh_y_d[bh] for bh in bh_k_l]


        x_title = measure_name
        if measure_name == 'duration':
            x_title += ' (s)'

        new_bh_k_l = list(bh_y_d.keys())

        def test_func(x, N, b):
            return N*(np.exp(-(x/b)))

        for i, bh in enumerate(bh_k_l):
            print(x_l[i])
            print(bh_y_l[i])
            params, params_covariance = optimize.curve_fit(test_func, x_l[i], bh_y_l[i])
            y_fit = test_func(x_l[i], *params)

            par = [v for v in params]
            par.insert(0, bh)
            new_bh_k_l.append('{}__{:.1e}*exp<sup>-(t/{:.1e})<sup>'.format(*par))
            bh_y_l.append(y_fit)
            x_l.append(x_l[i])

        return plotly_utils.plot_scatter_mult(x_l=x_l, y_l_l=bh_y_l,
                                                name_l=new_bh_k_l, title='{}s distribution'.format(measure_name),
                                                x_title=x_title, y_title='',
                                                xrange=(min_measure, max_measure), confidence=confidence, with_stats=True)


    def amplitude_distribution_plot_violin_duo(self, astroA_l, bh_1, bh_2, max_dff=5):
        amp_l_1 = []
        amp_l_2 = []

        for astroA in astroA_l:
            if bh_1 in astroA.event_subsets.keys():
                amp_l_1.extend(list(astroA.res_d['dffMax2'][astroA.event_subsets[bh_1]]))
            if bh_2 in astroA.event_subsets.keys():
                amp_l_2.extend(list(astroA.res_d['dffMax2'][astroA.event_subsets[bh_2]]))

        amp_l_1 = np.array(amp_l_1)
        amp_l_2 = np.array(amp_l_2)

        if max_dff is not None:
            amp_l_1 = amp_l_1[amp_l_1 <= max_dff]
            amp_l_2 = amp_l_2[amp_l_2 <= max_dff]

        fig = plotly_utils.plot_violin_duo(bh_1, bh_2, amp_l_1, amp_l_2, title='', x_title='', y_title='')
        return fig

    def sizes_distribution_plot_violin_duo(self, astroA_l, bh_1, bh_2, max_area=18):
        sizes_l_1 = []
        sizes_l_2 = []

        for astroA in astroA_l:
            if bh_1 in astroA.event_subsets.keys():
                sizes_l_1.extend(list(astroA.res_d['area'][astroA.event_subsets[bh_1]]))
            if bh_2 in astroA.event_subsets.keys():
                sizes_l_2.extend(list(astroA.res_d['area'][astroA.event_subsets[bh_2]]))

        sizes_l_1 = np.array(sizes_l_1)
        sizes_l_2 = np.array(sizes_l_2)

        if max_area is not None:
            sizes_l_1 = sizes_l_1[sizes_l_1 <= max_area]
            sizes_l_2 = sizes_l_2[sizes_l_2 <= max_area]

        fig = plotly_utils.plot_violin_duo(bh_1, bh_2, sizes_l_1, sizes_l_2, title='', x_title='', y_title='')
        return fig

    def signal_duration_distribution_plot_violin_duo(self, astroA_l, bh_1, bh_2, max_duration=100):
        durations_l_1 = []
        durations_l_2 = []

        for astroA in astroA_l:
            if bh_1 in astroA.event_subsets.keys():
                durations_l_1 = astroA.all_durations_d[bh_1]
            if bh_2 in astroA.event_subsets.keys():
                durations_l_2 = astroA.all_durations_d[bh_2]

        durations_l_1 = np.array(durations_l_1)
        durations_l_2 = np.array(durations_l_2)

        if max_duration is not None:
            durations_l_1 = durations_l_1[durations_l_1 <= max_duration]
            durations_l_2 = durations_l_2[durations_l_2 <= max_duration]

        fig = plotly_utils.plot_violin_duo(bh_1, bh_2, durations_l_1, durations_l_2, title='Signal duration (s) distribution', x_title='', y_title='')
        plotly_utils.apply_fun_axis_fig(fig, lambda x : x / astroA_l[0].fr, axis='y')
        return fig

    def get_stick_run_sample_figs(self, astroA):
        figs = []

        possible_spots = list(np.sort(list(set(astroA.indices_d['stick_exact_start']) & set(astroA.indices_d['running_exact']))))
        print(np.random.choice(possible_spots, min(len(possible_spots), 10), replace=False))
        np.random.seed(0)
        for spot in np.random.choice(possible_spots, min(len(possible_spots), 10), replace=False):
            time_from = spot - 100
            time_to = spot + 100

            print('time from to', time_from, time_to)

            stick_start_bin = np.zeros([len(astroA.stick_bin)])
            stick_start_bin[astroA.indices_d['stick_exact_start']] = 1

            stick_signal = stick_start_bin[time_from:time_to]
            #Obtain running signal
            running_signal = astroA.speed_values[time_from:time_to]

            fig_running = plotly_utils.plot_scatter_fmt(np.arange(len(running_signal)), running_signal, astype='float')
            fig_stick = plotly_utils.plot_scatter_fmt(np.arange(len(stick_signal)), stick_signal, astype='int')

            #Obtain available events during this period
            interval_events = list(set(np.where(astroA.res_d['tBegin'] > time_from)[0]) & set(np.where(astroA.res_d['tEnd'] < time_to)[0]))

            signal_figs = []
            for i, event_i in enumerate(interval_events[0:10]):
                adj_from = int(time_from % astroA.input_shape[2])
                adj_to = int(time_to % astroA.input_shape[2])

                if adj_to < adj_from:
                    print('Skipping: change time from to')
                    continue

                print('ADJ FROM {} ADJ TO {}'.format(adj_from, adj_to))
                y = astroA.res_d['dff_only'][event_i][adj_from:adj_to]
                x = np.arange(0, adj_to-adj_from)

                adj_begin = int(astroA.res_d['tBegin'][event_i] % astroA.input_shape[2]) - adj_from
                adj_end = int(astroA.res_d['tEnd'][event_i] % astroA.input_shape[2]) - adj_from

                print(adj_begin, adj_end)
                fig = plotly_utils.plot_scatter_signal(x, y, adj_begin, adj_end, mode='lines', title='scatter', x_title='', y_title='', with_legend=False)
                signal_figs.append(fig)

            figs.append([fig_running, fig_stick, signal_figs])
        return figs



    def get_compare_align_plot_xcorr_all(self, astro_pair_l, align_setting='xcorr', dff_mode=False, behaviour='default', filter_duration=(None, None),
                                            with_border_align=True, n_fake_samples=5, save_results_path=None):
        '''
        Go with each astrocyte pairs
            Calculate day 0-day x correlation
            Calculate random samples correlation
            Normalize s.t. random samples correlation for all pairs is the same (and the 0-x corr)
            Create plot

        '''

        pair_fakes = []
        pair_corrs_l = []
        days_id_l = []
        for astro_pair in astro_pair_l:
            astro_1, astro_2 = astro_pair[0], astro_pair[1]
            days = (str(astro_pair[0].day), str(astro_pair[1].day))
            days_id = '-'.join(days)

            print('DAYS', days)
            pair_save_results_path = save_results_path + self.get_astro_pair_id(astro_pair) + '.pkl'
            print('PAIR SAVE REUSLT PATH' , pair_save_results_path)

            if os.path.isfile(pair_save_results_path):
                print('FILE EXISTS')
                d = saving_utils.load_pickle(pair_save_results_path)
            else:
                if align_setting == 'xcorr':
                    #Get move vector
                    move_vector = compare_astro_utils.get_move_vector_xcorr_default(astro_1, astro_2)

                    #self.n_samples_corr_fake
                    d = compare_astro_utils.alignment_counter(astro_1, astro_2,
                                                                n_fake_samples=n_fake_samples,
                                                                align_setting='param',
                                                                eval_setting='xcorr',
                                                                fake_sample_setting='from_astro',
                                                                move_vector=move_vector,
                                                                p=1,
                                                                behaviour=behaviour,
                                                                filter_duration=filter_duration,
                                                                with_output_details=True)
                elif align_setting == 'xcorr_free':
                    d = compare_astro_utils.alignment_counter(astro_1, astro_2,
                                                                n_fake_samples=n_fake_samples,
                                                                align_setting='param',
                                                                eval_setting='xcorr_free',
                                                                fake_sample_setting='from_astro',
                                                                move_vector=None,
                                                                p=1,
                                                                behaviour=behaviour,
                                                                filter_duration=filter_duration,
                                                                with_output_details=True)
                if save_results_path is not None:
                    saving_utils.save_pickle(d, pair_save_results_path)

            print(d)
            pair_fakes.append(d['num_fake'])
            pair_corrs_l.append(d['num_compare'])
            days_id_l.append(days_id)

            print('DAYS: {} FAKE {} COMPARE {}:', days_id, np.mean(d['num_fake']), np.mean(d['num_compare']))
            print('all fake vals:', d['num_fake'])

        pair_fakes_before = np.copy(pair_fakes)
        pair_corrs_l_before = np.copy(pair_corrs_l)

        #print('PAIR FAKES', pair_fakes)
        mean_num_fake = np.mean([np.mean(pair_fake) for pair_fake in pair_fakes])
        print('mean num fake:', mean_num_fake)
        pair_corrs_d = {}
        for i in range(len(pair_corrs_l)):
            print('Scaling {}'.format(days_id_l[i]))
            #mult = mean_num_fake / np.mean(pair_fakes[i])
            #NOT DOING ANY SCALING
            mult = 1
            print('before pair corrs', pair_corrs_l[i])
            print('before pair fakes', pair_fakes[i])
            pair_fakes[i] = np.array(pair_fakes[i]) * mult
            pair_corrs_l[i] = pair_corrs_l[i] * mult
            print('after pair corrs', pair_corrs_l[i])
            print('after pair fakes', pair_fakes[i])
            if days_id_l[i] not in pair_corrs_d:
                pair_corrs_d[days_id_l[i]] = []
            pair_corrs_d[days_id_l[i]].append(pair_corrs_l[i])

        print('Pair corrs', pair_corrs_d)


        x = ['fake_samples']
        y = [[item for sublist in pair_fakes for item in sublist]]

        for k in pair_corrs_d.keys():
            x.append('days ' + k)
            y.append(pair_corrs_d[k])

        #tstat, pvalue = ttest_ind_from_stats(np.mean(y[0]), np.std(y[0]), len(y[0]), np.mean(y[1]), np.std(y[1]), len(y[1]))
        #print('NUM COMPARE: {}, mode {} behaviour {}'.format(d['num_compare'], dff_mode, behaviour))
        fig = plotly_utils.plot_point_box_revised(x, y, title='Behaviour: {} - correlations'.format(behaviour), x_title='', y_title='Aligned xcorr value')

        return fig, pair_fakes_before, pair_fakes, pair_corrs_l_before, pair_corrs_l, days_id_l

    def get_compare_states_all_xcorr(self, astro_pair, align_setting='xcorr_free', dff_mode='False', n_fake_samples=5, save_pkl_path=None, filter_duration=(None, None),
        behaviour_l=['rest', 'running', 'stick_rest', 'stick_run_ind_15', 'default']):
        astro_1, astro_2 = astro_pair

        print('Working on {}'.format(self.get_astro_pair_id(astro_pair)))
        if os.path.isfile(save_pkl_path):
            print('FILE EXISTS')
            res_d = saving_utils.load_pickle(save_pkl_path)
        else:
            res_d = {}
            for behaviour in behaviour_l:
                print('Current behaviour: ', behaviour)
                if (behaviour in astro_1.indices_d) and (behaviour in astro_2.indices_d) and \
                    (behaviour in astro_1.event_subsets) and (behaviour in astro_2.event_subsets):

                    if align_setting == 'xcorr':
                        #Get move vector
                        move_vector = compare_astro_utils.get_move_vector_xcorr_default(astro_1, astro_2)

                        #self.n_samples_corr_fake
                        d = compare_astro_utils.alignment_counter(astro_1, astro_2,
                                                                    n_fake_samples=n_fake_samples if behaviour == 'default' else 0,
                                                                    align_setting='param',
                                                                    eval_setting='xcorr',
                                                                    fake_sample_setting='from_astro',
                                                                    move_vector=move_vector,
                                                                    p=1,
                                                                    behaviour=behaviour,
                                                                    filter_duration=filter_duration,
                                                                    with_output_details=True,
                                                                    dff_mode=dff_mode)
                    elif align_setting == 'xcorr_free':
                        d = compare_astro_utils.alignment_counter(astro_1, astro_2,
                                                                    n_fake_samples=n_fake_samples if behaviour == 'default' else 0,
                                                                    align_setting='param',
                                                                    eval_setting='xcorr_free',
                                                                    fake_sample_setting='from_astro',
                                                                    move_vector=None,
                                                                    p=1,
                                                                    behaviour=behaviour,
                                                                    filter_duration=filter_duration,
                                                                    with_output_details=True,
                                                                    dff_mode=dff_mode)
                    print('NU COMPARE', d['num_compare'])
                    res_d[behaviour] = d['num_compare']

                    if behaviour == 'default':
                        res_d['random'] = d['num_fake']
                else:
                    print('Behaviour {} not in one of {} / {}'.format(behaviour, astro_1.id, astro_2.id))
        print('RES D', self.get_astro_pair_id(astro_pair),  res_d)
        if save_pkl_path is not None:
            saving_utils.save_pickle(res_d, save_pkl_path)

        behaviours = [b for b in behaviour_l]
        behaviours.append('random')
        x = []
        y = []
        print('RES D', res_d)
        for k in behaviours:
            if ((k in astro_1.indices_d) and (k in astro_2.indices_d) and (k in astro_1.event_subsets) and (k in astro_2.event_subsets)) or (k=='random'):
                if k != 'random':
                    res_d[k] = [res_d[k]]
                x.append(k)
                y.append(res_d[k])
        #x = behaviour_l
        #y = [res_d['rest'], res_d['running'], res_d['default'], res_d['random']]

        print(y)
        fig = plotly_utils.plot_point_box_revised(x, y, title='Behaviour correlations', x_title='Behaviour', y_title='Xcorr value')

        return fig, res_d

    def get_compare_states_same_astro_all_xcorr(self, astro_pair, align_setting='xcorr_free', dff_mode=False, n_fake_samples=5, save_pkl_path=None, filter_duration=(None, None)):
        if os.path.isfile(save_pkl_path):
            print('FILE EXISTS')
            res_d = saving_utils.load_pickle(save_pkl_path)
        else:
            res_d = {}
            astro_1, astro_2 = astro_pair

            for astro in astro_pair:
                astro_day = astro.day
                for behaviour_pair in [['rest', 'running'], ['default', 'default']]:
                    astro_a_grid, _, _,_ = compare_astro_utils.get_filters_compare([astro], p=1, dff_mode=dff_mode, behaviour=behaviour_pair[0])
                    astro_a_grid = astro_a_grid[0]
                    astro_b_grid, _, _,_ = compare_astro_utils.get_filters_compare([astro], p=1, dff_mode=dff_mode, behaviour=behaviour_pair[1])
                    astro_b_grid = astro_b_grid[0]
                    if align_setting == 'xcorr':
                        #Get move vector
                        move_vector = compare_astro_utils.get_move_vector_xcorr_default(astro_1, astro_2)

                        d = compare_astro_utils.alignment_counter(astro_1, astro_2,
                                                                    n_fake_samples=n_fake_samples if behaviour_pair[0] == 'default' else 0,
                                                                    align_setting='param',
                                                                    eval_setting='xcorr',
                                                                    fake_sample_setting='from_astro',
                                                                    grid_target=astro_a_grid,
                                                                    grid_source=astro_b_grid,
                                                                    move_vector=move_vector,
                                                                    p=1,
                                                                    behaviour='default',
                                                                    filter_duration=filter_duration,
                                                                    with_output_details=True)
                    elif align_setting == 'xcorr_free':
                        d = compare_astro_utils.alignment_counter(astro_1, astro_2,
                                                                    n_fake_samples=n_fake_samples if behaviour_pair[0] == 'default' else 0,
                                                                    align_setting='param',
                                                                    eval_setting='xcorr_free',
                                                                    fake_sample_setting='from_astro',
                                                                    grid_target=astro_a_grid,
                                                                    grid_source=astro_b_grid,
                                                                    move_vector=None,
                                                                    p=1,
                                                                    behaviour='default',
                                                                    filter_duration=filter_duration,
                                                                    with_output_details=True)
                    print('NU COMPARE', d['num_compare'])

                    if behaviour_pair[0] == 'rest':
                        res_d['_'.join(behaviour_pair) + '_{}'.format(astro_day)] = d['num_compare']
                    if behaviour_pair[0] == 'default':
                        res_d['random_{}'.format(astro_day)] = d['num_fake']

        print('RES D', self.get_astro_pair_id(astro_pair),  res_d)
        if save_pkl_path is not None:
            saving_utils.save_pickle(res_d, save_pkl_path)

        for k in res_d.keys():
            if 'random' not in k:
                res_d[k] = [res_d[k]]

        x = [k for k in res_d.keys()]
        y = [res_d[k] for k in x]
        #y = [res_d['rest'], res_d['running'], res_d['default'], res_d['random']]
        print(y)
        print(x)
        print('RES DHERE', res_d)
        fig = plotly_utils.plot_point_box_revised(x, y, title='Behaviour correlations', x_title='Behaviour', y_title='Xcorr value')

        return fig, res_d

    #TODO NOT DONE
    def get_compare_between_group_xcorr(self, astroA_l_pairs, n_fake_samples=5, dff_mode=False, save_pkl_path=None, filter_duration=[None, None]):
        if os.path.isfile(save_pkl_path):
            print('FILE EXISTS')
            res_d = saving_utils.load_pickle(save_pkl_path)
        else:
            res_d = {'between' : [], 'random' : [], 'between_id' : []}

            for astro_i in range(len(astroA_l_pairs)):
                for astro_j in range(astro_i+1, len(astroA_l_pairs)):
                    astroA_pair_1 = astroA_l_pairs[astro_i]
                    astroA_pair_2 = astroA_l_pairs[astro_j]
                    #quick hack, ignore the bad dataset
                    if astroA_pair_1[0].id == 'm190129_d190226_cx_day_0' or astroA_pair_2[0].id == 'm190129_d190226_cx_day_0':
                        continue
                    #continue if we are on same pair
                    if astroA_pair_1[0].id == astroA_pair_2[0].id:
                        continue

                    print('ASTRO PAIRS {}, {}'.format(self.get_astro_pair_id(astroA_pair_1), self.get_astro_pair_id(astroA_pair_2)))
                    for i in [0, 1]:
                        for j in [0, 1]:
                            print('I, J', i, j)
                            astro_pair = [astroA_pair_1[i], astroA_pair_2[j]]

                            d = compare_astro_utils.alignment_counter(astro_pair[0], astro_pair[1],
                                                                            n_fake_samples=n_fake_samples,
                                                                            align_setting='xcorr',
                                                                            eval_setting='xcorr_random_both',
                                                                            fake_sample_setting='from_astro',
                                                                            p=1,
                                                                            behaviour='default',
                                                                            dff_mode=dff_mode,
                                                                            border_nan=True,
                                                                            with_output_details=True)
                            res_d['between_id'].append(self.get_astro_pair_id(astro_pair))
                            res_d['between'].append(d['num_compare'])
                            res_d['random'].extend(d['num_fake'])

            if save_pkl_path is not None:
                saving_utils.save_pickle(res_d, save_pkl_path)
        print('RES D', res_d)
        x = ['Astro between group', 'Random between group']
        y = [res_d['between'], res_d['random']]
        fig = plotly_utils.plot_point_box_revised(x, y, title='Between group correlations vs random (95% confidence)', x_title='', y_title='Xcorr value')

        return fig, res_d

    def get_astro_pair_id(self, astro_pair):
        return '_'.join([astro.print_id for astro in astro_pair])


    def get_measure_all_bar_plot(self, astroA_l, measure, bh_list=['rest', 'running']):
        y_pair_l = [[] for i in range(len(bh_list))]
        err_pair_l = [[] for i in range(len(bh_list))]
        length_l = [[] for i in range(len(bh_list))]
        x = []
        for astroA in astroA_l:
            x.append(astroA.print_id)
            for i, bh in enumerate(bh_list):
                measure_res = astroA.res_d[measure][astroA.event_subsets[bh]]
                mean, conf_low, conf_high = stat_utils.mean_confidence_interval(measure_res, confidence=0.95)

                conf = conf_high - mean
                y_pair_l[i].append(mean)
                err_pair_l[i].append(conf)
                length_l[i].append(len(measure_res))

        fig = plotly_utils.plot_group_bar(x, y_pair_l, text_values_l=length_l, title='', text_size=20, x_title='', y_title='', legends=bh_list, std_y=err_pair_l, margin_b=300, margin_r=300)
        return fig

    def get_measure_all_dot_plot(self, astroA_l, measure, bh_list=['rest', 'running']):
        x_l = bh_list
        name_l=[]
        y_pair_l_l = []
        for astroA in astroA_l:
            name_l.append(astroA.print_id)
            y_pair_l = [[] for i in range(len(bh_list))]
            length_l = [[] for i in range(len(bh_list))]

            for i, bh in enumerate(bh_list):
                if measure != None:
                    measure_res = astroA.res_d[measure][astroA.event_subsets[bh]]
                    y_pair_l[i].append(measure_res)
                    print(astroA.print_id, 'state', bh, 'measure', measure, np.mean(measure_res))
                else:
                    n = (len(astroA.event_subsets[bh]) / len(astroA.indices_d[bh])) * astroA.minute_frames
                    print('BH {} n {}'.format(bh, n))
                    y_pair_l[i].append([n])
            y_pair_l_l.append(y_pair_l)
        fig, stats_d = plotly_utils.plot_scatter_mult_with_avg(x_l=x_l, y_l_l=y_pair_l_l, y_mean=None, name_l=name_l, mode='lines', x_title='', y_title='',
                                    confidence=True, with_stats=True)

        return fig, stats_d


    def get_before_after_transition_events(self, astroA, before_bh, inds_bh, after_bh, before_range=20, after_range=50, measure=None,
                                            duration_filter=[None, None]):
        inds = astroA.indices_d[inds_bh]
        #Filter indices
        indices_filt_before = aqua_utils.filter_range_inds(inds, astroA.indices_d[before_bh], range=(-before_range, -1), prop=1.0)
        indices_filt_after = aqua_utils.filter_range_inds(inds, astroA.indices_d[after_bh], range=(1, after_range), prop=1.0)
        indices_filt = np.array(np.sort(list(set(indices_filt_before) & set(indices_filt_after))))

        #print('LEN INDICES_FILT: {}'.format(len(indices_filt)))
        #print('TOTAL IND {} BEFORE {} AFTER {} JOIN {}'.format(len(inds), len(indices_filt_before), len(indices_filt_after), len(indices_filt)))
        if len(indices_filt) == 0:
            return [], []
        #print('Len indices {} len filt before {} len filt after {} len filt {}'.format(len(inds), len(indices_filt_before), len(indices_filt_after), len(indices_filt)))
        #print('LEN INDICES FILT : {}'.format(len(indices_filt)))

        delay_info_args = {'event_inds_subset' : astroA.event_subsets['default'],
                       'min_delay' : -before_range,
                       'max_delay' : after_range,
                       'min_duration' : duration_filter[0],
                       'max_duration' : duration_filter[1],
                       'unique_events' : False,
                       'return_non_unique_delays_arr' : True
                       }

        _, _, _, signal_delays_l_l, peak_mins_l_l, valid_event_i_l_l = aqua_utils.get_delay_info_from_res(indices_filt, astroA.res_d, **delay_info_args)

        print('Num unique indices CHECK {}'.format(len(valid_event_i_l_l)))
        if measure is None:
            before_l = 0
            after_l = 0
        else:
            before_l = []
            after_l = []

        for i, signal_delays_l in enumerate(signal_delays_l_l):
            signal_delays_np = np.array(signal_delays_l)
            if measure is None:
                before_l += len(signal_delays_np[signal_delays_np < 0])
                after_l  += len(signal_delays_np[signal_delays_np > 0])
            else:
                measure_np = np.array(list(astroA.res_d[measure][valid_event_i_l_l[i]]))
                before_l.extend(list(measure_np[signal_delays_np < 0]))
                after_l.extend(list(measure_np[signal_delays_np > 0]))

        if measure is None:
            before_l = [before_l]
            after_l = [after_l]

        return before_l, after_l

    def get_measure_all_transition_dot_plot(self, astroA_l, measure, before_bh, inds_bh,
                                            after_bh, before_range=20, after_range=50, duration_filter=[None, None]):
        '''
        In get measure all dot plot we take a list of behaviours : e.g. [rest, running]

        Then we find the events that take place during each behaviour
        Then we either measure number of events normalized to minute or the measure values

        Here we care about transition. We first find all events that are before transition and then after transition

        '''
        x_l = [before_bh + '-' + inds_bh, inds_bh + '-' + after_bh]
        name_l=[]
        y_pair_l_l = []
        for astroA in astroA_l:
            name_l.append(astroA.print_id)
            y_pair_l = [[] for i in range(2)]
            length_l = [[] for i in range(2)]

            #Find events or number of events for behaviour before and after

            before_l, after_l = self.get_before_after_transition_events(astroA, before_bh, inds_bh, after_bh,
                                                                        before_range=before_range, after_range=after_range,
                                                                        measure=measure, duration_filter=duration_filter)

            y_pair_l[0].append(before_l)
            y_pair_l[1].append(after_l)

            y_pair_l_l.append(y_pair_l)
        fig, stats_d = plotly_utils.plot_scatter_mult_with_avg(x_l=x_l, y_l_l=y_pair_l_l, y_mean=None, name_l=name_l, mode='lines', x_title='', y_title='',
                                    confidence=True, with_stats=True)

        return fig, stats_d

    def duration_size_amplitude_plot(self, astroA):
        areas = astroA.res_d['area']
        amplitudes = astroA.res_d['dffMax2']
        times = astroA.res_d['time_s']

        trace_i = go.Scatter(
            x=x_l[i],
            y=mean_l_l[i],
            mode=mode,
            name=name_l[i],
            line=dict(color=colour_l[i])
        )

        layout = go.Layout(title=title, xaxis=dict(title=x_title),
                            yaxis=dict(title=y_title),)

        if yrange is not None:
            layout.update(yaxis=dict(range=yrange))
        if xrange is not None:
            layout.update(xaxis=dict(range=xrange))

        fig = go.Figure(data=traces_l, layout=layout)
        '''
        plot_scatter_mult_tree(x, y_main, y_mult, mode_main='lines', mode_mult='markers',
                                    title='', y_title='', x_title='', fit=False, fit_annotation_pos_fix=1,
                                    bin_main_size=1, bin_mult_size=1, opacity=0.1, confidence=False, with_stats=False,
                                    y_mult_include=True, confidence_format='lines', bin_type='mean'):
        '''


    def triplet_bar_plot(self, astroA, bh='default', measure=None, n_bins=8, y_title=''):
        border_mask= astroA.border
        clandmark_center = astroA.res_d['clandmark_center']
        event_distances_from_center_micrometers = astroA.res_d['clandmark_distAvg'][astroA.event_subsets[bh]]
        event_distances_from_center = event_distances_from_center_micrometers / astroA.spatial_res

        n_events_arr_norm, n_events_i_arr, area_bins, r_bins = aqua_utils.radius_event_extraction(event_distances_from_center, clandmark_center, border_mask, n_bins=n_bins)

        y = []
        x = []
        err_l = []
        text_values_l = []

        if measure is None:
            #Y axis: number of events / (Area of band x time) -> time is to calibrate the number of events to a minute. X axis: radius of band from the center.
            for i, events_i in enumerate(n_events_i_arr):
                num_events = len(events_i)
                area = area_bins[i]

                #pp = per pixel
                num_events_pp = num_events / area
                #pm = per minute
                num_events_pp_pm = (num_events_pp / len(astroA.indices_d[bh])) * astroA.minute_frames
                #we have events per pixel, now we scale to events as if scaled up to be size of whole astrocyte
                print('sum area bins', np.sum(area_bins))
                num_events_pp_pm_norm_whole = num_events_pp_pm * np.sum(area_bins)
                y.append(num_events_pp_pm_norm_whole)
                x.append(r_bins[i+1])
                text_values_l.append('n={}'.format(len(events_i)))

                y_title='Events scaled to whole astrocyte size'
        else:
            #Y axis: mean duration (s). X axis: radius of band from the center
            #Y axis: mean size (um^2). X axis: radius of band from the center
            #Y axis amplitude (df/f). X axis: radius of band from center
            event_durations = astroA.res_d[measure][astroA.event_subsets[bh]]
            for i, events_i in enumerate(n_events_i_arr):
                event_durations_i = event_durations[events_i]
                ev_mean, ev_low, ev_high = stat_utils.mean_confidence_interval(event_durations_i, confidence=0.95)

                y.append(ev_mean)
                x.append(r_bins[i+1])
                err_l.append(ev_high - ev_mean)
                text_values_l.append('n={}'.format(len(events_i)))
                y_title=y_title
        fig = plotly_utils.plot_bar(x, y, err_y=err_l, text_size=20, text_values=text_values_l, y_title=y_title, x_title='Radius')
        return fig


    def triplet_dot_plot_all(self, astroA_l, bh='default', measure=None, n_bins=8, y_title=''):
        x_l_l = []
        y_l_l = []
        name_l = []

        astroA_l = [astroA for astroA in astroA_l if bh in astroA.event_subsets]
        y_mean_np=np.zeros([len(astroA_l), n_bins])

        for astroA_i, astroA in enumerate(astroA_l):
            border_mask= astroA.border
            clandmark_center = astroA.res_d['clandmark_center']
            event_distances_from_center_micrometers = astroA.res_d['clandmark_distAvg'][astroA.event_subsets[bh]]
            event_distances_from_center = event_distances_from_center_micrometers / astroA.spatial_res

            n_events_arr_norm, n_events_i_arr, area_bins, r_bins = aqua_utils.radius_event_extraction(event_distances_from_center, clandmark_center, border_mask, n_bins=n_bins)

            y_l = []
            x_l = []
            y_mean_l = []

            if measure is None:
                #Y axis: number of events / (Area of band x time) -> time is to calibrate the number of events to a minute. X axis: radius of band from the center.
                for i, events_i in enumerate(n_events_i_arr):
                    num_events = len(events_i)
                    area = area_bins[i]

                    #pp = per pixel
                    num_events_pp = num_events / area
                    #pm = per minute
                    num_events_pp_pm = (num_events_pp / len(astroA.indices_d[bh])) * astroA.minute_frames
                    #we have events per pixel, now we scale to events as if scaled up to be size of whole astrocyte
                    num_events_pp_pm_norm_whole = num_events_pp_pm * np.sum(area_bins)

                    y_l.append([num_events_pp_pm_norm_whole])
                    x_l.append(r_bins[i+1])

                    y_mean_np[astroA_i, i] = np.mean(num_events_pp_pm_norm_whole)
            else:
                #Y axis: mean duration (s). X axis: radius of band from the center
                #Y axis: mean size (um^2). X axis: radius of band from the center
                #Y axis amplitude (df/f). X axis: radius of band from center
                event_durations = astroA.res_d[measure][astroA.event_subsets[bh]]
                for i, events_i in enumerate(n_events_i_arr):
                    event_durations_i = event_durations[events_i]
                    y_l.append(event_durations_i)
                    x_l.append(r_bins[i+1])
                    y_title=y_title
                    y_mean_np[astroA_i, i] = np.mean(event_durations_i)

            x_l_l.append(x_l)
            y_l_l.append(y_l)
            name_l.append(astroA.print_id)

        y_mean_np = np.mean(y_mean_np, axis=0)
        x_l = ['Band {}'.format(i) for i in range(1, 1+len(x_l))]
        if measure is None:
            y_title='Events scaled to whole astrocyte size'
            fig, stats_d = plotly_utils.plot_scatter_mult_with_avg(x_l=x_l, y_l_l=y_l_l, y_mean=list(y_mean_np), name_l=name_l, mode='lines', x_title='Band (0-{})'.format(n_bins-1), y_title=y_title,
                                    confidence=False, avg_confidence=True, with_stats=True)
        else:
            y_title=y_title
            fig, stats_d = plotly_utils.plot_scatter_mult_with_avg(x_l=x_l, y_l_l=y_l_l, y_mean=list(y_mean_np), name_l=name_l, mode='lines', x_title='Band (0-{})'.format(n_bins-1), y_title=y_title,
                                    confidence=True, with_stats=True)

        return fig, stats_d
