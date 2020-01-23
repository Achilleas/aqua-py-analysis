from importlib import reload  # Python 3.4+ only.
import h5py
import os, sys, glob
import numpy as np
import plotly.offline as offline
from preprocessing import analysis_pp
from analysis.general_utils import aqua_utils, saving_utils, graph_utils, plotly_utils, general_utils, correlation_utils
from scipy.ndimage.filters import gaussian_filter
import csv
import math
from pandas import DataFrame

class AxonAstroPlotter():
    def __init__(self, output_folder):
        self.output_folder = output_folder
        self.n_signal_samples = 2
        self.start_delay_max_corr = 15

    def get_output_experiment_path(self, AA):
        experiment_id = '/'.join(AA.experiment_path.split('/')[-3:])
        output_experiment_path = os.path.join(self.output_folder, experiment_id)
        return output_experiment_path

    def setup_folders(self, output_experiment_path):
        paths = ['landmark_borders',
                 'ddr_heatmaps',
                 'behaviours_basic',
                 'behaviour_heatmaps',
                 'behaviour_activity',
                 'events_after_count_csvs',
                 'delays_per_landmark_csvs',
                 'events_after_count_plots',
                 'signal_basic_samples',
                 'signal_landmark_samples',
                 'axon_stick_corr_plots',
                 'axon_far_after_stick_signals',
                 'axon_border_after_stick_signals',
                 'landmark_distances',
                 'run_stick_run_axon_proportions',
                 'rest_to_run_axon_proportions',
                 'rest_to_run_astro_proportions',
                 'rest_to_run_STICK_FILT_astro_proportions',
                 'rest_to_run_STICK_FILT_axon_proportions',
                 'run_to_rest_axon_proportions',
                 'run_to_rest_astro_proportions',
                 'run_to_rest_STICK_FILT_astro_proportions',
                 'run_to_rest_STICK_FILT_axon_proportions',
                 'run_stick_run_axon_proportions',
                 'run_stick_run_astro_proportions',
                 'run_stick_run_STICK_FILT_astro_proportions',
                 'run_stick_run_STICK_FILT_axon_proportions']

        for p in paths:
            try:
                os.makedirs(os.path.join(output_experiment_path, 'plots' , p))
            except:
                pass

    def setup_plot_folders_all(self, output_experiment_path_comparison):
        paths = ['correlation_results', 'correlation_results_inverse', 'correlation_signal_samples',
                'correlation_signal_samples_inverse',
                'signal_delays_axon', 'signal_delays_axon_border',
                'prop_signal_delays_axon_border', 'prop_signal_delays_axon_border_avg',
                'prop_signal_delays_axon_border_gauss_filter',
                'waterfall_together',
                'rest_to_run_axon_proportions',
                'rest_to_run_astro_proportions',
                'rest_to_run_STICK_FILT_astro_proportions',
                'rest_to_run_STICK_FILT_axon_proportions',
                'run_to_rest_axon_proportions',
                'run_to_rest_astro_proportions',
                'run_to_rest_STICK_FILT_astro_proportions',
                'run_to_rest_STICK_FILT_axon_proportions',
                'run_stick_run_axon_proportions',
                'run_stick_run_astro_proportions',
                'run_stick_run_STICK_FILT_astro_proportions',
                'run_stick_run_STICK_FILT_axon_proportions',

                'rest_to_run_speed',
                'run_to_rest_speed',
                'run_stick_run_speed',]

        data_paths = [
            'behaviour_ratios',
        ]

        for p in data_paths:

            try:
                os.makedirs(os.path.join(output_experiment_path_comparison, 'data', p))
            except:
                print('Folder structure exists?')

        for p in paths:
            try:
                os.makedirs(os.path.join(output_experiment_path_comparison, 'plots', p))
            except:
                print('Folder structure exists?')

    def setup_comparison_vars(self, astroA_l):
        experiment_id_l = []
        day_l = []
        for astroA in astroA_l:
            experiment_id_l.append('/'.join(astroA.experiment_path.split('/')[-3:-1]))
            day_l.append(int(astroA.experiment_path.split('/')[-1].split('_')[-1]))

        sort_i = np.argsort(day_l)
        day_l_s = [day_l[i] for i in sort_i]
        astroA_l_s = [astroA_l[i] for i in sort_i]

        combine_str = '-'.join([astroA.print_id for astroA in astroA_l_s])
        output_experiment_path_comparison = os.path.join(self.output_folder,
                                                        experiment_id_l[0],
                                                        combine_str)
        print('done')
        return output_experiment_path_comparison, combine_str, day_l_s, astroA_l_s

    def plot_all_single(self, AA):
        output_experiment_path = self.get_output_experiment_path(AA)
        print('Making dirs', output_experiment_path)
        self.setup_folders(output_experiment_path)

        '''
        print('Creating .csv file of distances between landmarks...')
        landmark_distances_path = os.path.join(output_experiment_path, 'plots', 'landmark_distances', 'distances.csv')
        self.create_landmark_distances_csv(AA, landmark_distances_path)
        '''
        '''
        print('Plotting behaviours basic...')
        #Behaviour basic
        fig_stick, fig_speed, fig_pupil = self.get_behaviour_basic_plots(AA)
        saving_utils.save_plotly_fig(fig_stick, os.path.join(output_experiment_path, 'plots', 'behaviours_basic', 'stick'), width=1000, height=400)
        saving_utils.save_plotly_fig(fig_speed, os.path.join(output_experiment_path, 'plots', 'behaviours_basic', 'speed'), width=1000, height=400)
        saving_utils.save_plotly_fig(fig_pupil, os.path.join(output_experiment_path, 'plots', 'behaviours_basic', 'pupil'), width=1000, height=800)
        '''
        '''
        print('Plotting random samples of signals...')
        fig_signals = self.get_signal_figs_samples(AA, 20)
        for i, fig_signal in enumerate(fig_signals):
            fig_signal_path = os.path.join(output_experiment_path, 'plots', 'signal_basic_samples', 'signal_{}'.format(i))
            saving_utils.save_plotly_fig(fig_signal, fig_signal_path)

        '''
        '''
        print('Plotting landmark samples of signals...')
        fig_landmark_signals = self.get_signal_landmark_figs_samples(AA, 10)

        for bk in fig_landmark_signals.keys():
            for lk in fig_landmark_signals[bk].keys():
                for fig_i, fig in enumerate(fig_landmark_signals[bk][lk]):
                    if fig is None:
                        continue
                    fig_signal_path = os.path.join(output_experiment_path, 'plots', 'signal_landmark_samples', 'signal_{}-{}-{}'.format(bk, lk, fig_i))
                    saving_utils.save_plotly_fig(fig, fig_signal_path)
        '''
        '''
        print('Plotting behaviour heatmaps...')
        #Behaviour heatmaps
        fig_heatmap_grids, fig_heatmap_dff_grids = self.get_behaviour_contour_plots(AA)
        heatmap_grid_base_path = os.path.join(output_experiment_path, 'plots', 'behaviour_heatmaps')

        for k in fig_heatmap_grids.keys():
            saving_utils.save_plotly_fig(fig_heatmap_grids[k], os.path.join(heatmap_grid_base_path, k))
            saving_utils.save_plotly_fig(fig_heatmap_dff_grids[k], os.path.join(heatmap_grid_base_path, k + 'dff'))


        print('Plotting behaviour activity bar plot...')
        behaviour_activity_path = os.path.join(output_experiment_path, 'plots', 'behaviour_activity', 'activity')
        fig_behaviour_activity = self.get_behaviour_activity_plot(AA)
        saving_utils.save_plotly_fig(fig_behaviour_activity, behaviour_activity_path, width=1200, height=800)
        '''
        '''
        #print('Plotting ddr heatmaps...')
        #ddr_heatmap_base_path = os.path.join(output_experiment_path, 'plots', 'ddr_heatmaps')
        #figs_drr_heatmaps = self.get_ddr_heatmap_figs(AA)

        #for bk in figs_drr_heatmaps.keys():
        #    for axon_k in figs_drr_heatmaps[bk].keys():
        #        try:
        #            saving_utils.save_plotly_fig(figs_drr_heatmaps[bk][axon_k], os.path.join(ddr_heatmap_base_path, bk + '_' + axon_k), width=1000, height=1000)
        #        except:
        #            print(bk + ' _fail')
        #            pass
        '''
        '''
        print('Plotting landmark borders...')
        landmark_borders_base_path = os.path.join(output_experiment_path, 'plots', 'landmark_borders')
        figs_landmark_borders, figs_landmark_lines = self.get_landmark_border_figs(AA)

        for lk in figs_landmark_borders.keys():
            saving_utils.save_plotly_fig(figs_landmark_borders[lk], os.path.join(landmark_borders_base_path, lk + '_borders'), width=1000, height=1000)
            saving_utils.save_plotly_fig(figs_landmark_lines[lk], os.path.join(landmark_borders_base_path, lk + '_lines'), width=1000, height=1000)

        print('Events after counts plots...')
        events_after_count_plot_base_path = os.path.join(output_experiment_path, 'plots', 'events_after_count_plots')
        delays_after_count_plots = self.get_delays_after_count_plots(AA)
        for bk in AA.indices_d.keys():
            for axon_str in AA.axon_strs:
                saving_utils.save_plotly_fig(delays_after_count_plots[bk][axon_str], os.path.join(events_after_count_plot_base_path, bk + '_' + axon_str), width=1000, height=1000)

        print('Events after counts...')
        events_after_count_csv_path = os.path.join(output_experiment_path, 'plots', 'events_after_count_csvs')
        self.events_after_count_csvs(AA, events_after_count_csv_path)

        print('Delays per landmark csvs...')
        delays_per_landmark_csvs_path = os.path.join(output_experiment_path, 'plots', 'delays_per_landmark_csvs')
        self.delays_per_landmark_csvs(AA, delays_per_landmark_csvs_path)

        print('Stick stimulation axon correlation....')

        axon_stick_corr_figs = self.get_plots_whisker_correlations(AA, bh='default', delay_max=50)
        axon_stick_corr_base_path = os.path.join(output_experiment_path, 'plots', 'axon_stick_corr_plots')
        for k in axon_stick_corr_figs.keys():
            saving_utils.save_plotly_fig(axon_stick_corr_figs[k], os.path.join(axon_stick_corr_base_path, '{}'.format(k)))

        print('Axon FAR signals immediately after stick event...')
        stick_axon_event_figs = self.get_after_stick_axon_events_plot(AA, mode='axon_far')
        after_stick_axon_event_path = os.path.join(output_experiment_path, 'plots', 'axon_far_after_stick_signals')

        for i, fig_sub_l in enumerate(stick_axon_event_figs):
            for j, fig in enumerate(fig_sub_l):
                saving_utils.save_plotly_fig(fig, os.path.join(after_stick_axon_event_path, 'example_{}_{}'.format(i, j)))

        print('Axon Border (all axon only) signals immediately after stick event...')
        stick_axon_event_figs = self.get_after_stick_axon_events_plot(AA, mode='axon_border')
        after_stick_axon_event_path = os.path.join(output_experiment_path, 'plots', 'axon_border_after_stick_signals')

        for i, fig_sub_l in enumerate(stick_axon_event_figs):
            for j, fig in enumerate(fig_sub_l):
                saving_utils.save_plotly_fig(fig, os.path.join(after_stick_axon_event_path, 'example_{}_{}'.format(i, j)))
        '''

        """
        print('---TRANSITION PROPORTION DELAYS PLOT ALL--')
        delay_ranges_pairs = [ [3*AA.fr, 6*AA.fr],
                                [2*AA.fr, 4*AA.fr],
                               #[1*AA.fr, 1*AA.fr]]
                               ]
        delay_ranges_pairs = [[int(v[0]), int(v[1])] for v in delay_ranges_pairs]

        print('Alt Proportion plots...')
        #for setting in ['axon', 'astro']:
        for setting in ['axon', 'astro']:
            for delay_ranges_pair in delay_ranges_pairs:
                before_range, after_range = delay_ranges_pair
                for p in [#{'fit' : True, 'delay_step_size' : 1, 'confidence' : True},
                          #{'fit' : True, 'delay_step_size' : 5, 'confidence' : True},
                          {'fit' : True, 'delay_step_size' : 10, 'confidence': True, 'setting' : setting}
                          ]:

                    ################################################
                    ##############Proportion plots##################
                    ################################################

                    '''
                    print('EXTRA PARS', p, p.keys())

                    print('rest to run prop')
                    path = os.path.join(output_experiment_path, 'plots', 'rest_to_run_{}_proportions'.format(setting))
                    fig_d, bin_stats = self.get_axon_transition_proportion_delays_plot_all([AA], before_bh='rest_semi_exact', inds_bh='running_exact_start', after_bh='running_semi_exact',
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
                    path = os.path.join(output_experiment_path, 'plots', 'run_to_rest_{}_proportions'.format(setting))
                    fig_d, bin_stats = self.get_axon_transition_proportion_delays_plot_all([AA], before_bh='running_semi_exact', inds_bh='rest_start', after_bh='rest_semi_exact',
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
                    path = os.path.join(output_experiment_path, 'plots', 'run_stick_run_{}_proportions'.format(setting))
                    fig_d, bin_stats = self.get_axon_transition_proportion_delays_plot_all([AA], before_bh='running_semi_exact', inds_bh='stick_exact_start', after_bh='running_semi_exact',
                                                                                    before_range=before_range, after_range=after_range,
                                                                                    **p)
                    for fig_k in fig_d:
                        fig_id = os.path.join(path, fig_k + 'range_{}_{}-{}-fit_{}-step_{}-conf_{}'.format(before_range, after_range, fig_k, p['fit'], p['delay_step_size'], p['confidence']))
                        saving_utils.save_plotly_fig(fig_d[fig_k], fig_id)
                        saving_utils.save_csv_dict(bin_stats, path=fig_id + '.csv', key_order=['x', 'mean', 'std', 'confidence_95'])


                    if p['delay_step_size'] == 10:
                        data_csv_path = os.path.join(path, 'range_{}_{}-step_{}-all.csv'.format(before_range, after_range, p['delay_step_size']))
                        DataFrame(bin_stats['y_all'], columns=bin_stats['x']).to_csv(data_csv_path, index=False)
                    '''

                    print('rest to run prop, STICK FILTERED')
                    path = os.path.join(output_experiment_path, 'plots', 'rest_to_run_STICK_FILT_{}_proportions'.format(setting))
                    fig_d, bin_stats = self.get_axon_transition_proportion_delays_STICK_FILTER_plot_all([AA], before_bh='rest_semi_exact', inds_bh='running_exact_start', after_bh='running_semi_exact',
                                                                                    before_range=before_range, after_range=after_range,
                                                                                    **p)
                    for fig_k in fig_d:
                        fig_id = os.path.join(path, fig_k + 'range_{}_{}-{}-fit_{}-step_{}-conf_{}'.format(before_range, after_range, fig_k, p['fit'], p['delay_step_size'], p['confidence']))
                        saving_utils.save_plotly_fig(fig_d[fig_k], fig_id)
                        saving_utils.save_csv_dict(bin_stats, path=fig_id + '.csv', key_order=['x', 'mean', 'std', 'confidence_95'])


                    if p['delay_step_size'] == 10:
                        data_csv_path = os.path.join(path, 'range_{}_{}-step_{}-all.csv'.format(before_range, after_range, p['delay_step_size']))
                        DataFrame(bin_stats['y_all'], columns=bin_stats['x']).to_csv(data_csv_path, index=False)

                    print('run to rest prop, STICK FILTERED')
                    path = os.path.join(output_experiment_path, 'plots', 'run_to_rest_STICK_FILT_{}_proportions'.format(setting))
                    fig_d, bin_stats = self.get_axon_transition_proportion_delays_STICK_FILTER_plot_all([AA], before_bh='running_semi_exact', inds_bh='rest_start', after_bh='rest_semi_exact',
                                                                                    before_range=before_range, after_range=after_range,
                                                                                    **p)
                    for fig_k in fig_d:
                        fig_id = os.path.join(path, fig_k + 'range_{}_{}-{}-fit_{}-step_{}-conf_{}'.format(before_range, after_range, fig_k, p['fit'], p['delay_step_size'], p['confidence']))
                        saving_utils.save_plotly_fig(fig_d[fig_k], fig_id)
                        saving_utils.save_csv_dict(bin_stats, path=fig_id + '.csv', key_order=['x', 'mean', 'std', 'confidence_95'])


                    if p['delay_step_size'] == 10:
                        data_csv_path = os.path.join(path, 'range_{}_{}-step_{}-all.csv'.format(before_range, after_range, p['delay_step_size']))
                        DataFrame(bin_stats['y_all'], columns=bin_stats['x']).to_csv(data_csv_path, index=False)


                    '''
                    print('run stick hit run prop, STICK FILTERED')
                    path = os.path.join(output_experiment_path, 'plots', 'run_stick_run_STICK_FILT_{}_proportions'.format(setting))
                    fig_d, bin_stats = self.get_axon_transition_proportion_delays_STICK_FILTER_plot_all([AA], before_bh='running_semi_exact', inds_bh='stick_exact_start', after_bh='running_semi_exact',
                                                                                    before_range=before_range, after_range=after_range,
                                                                                    **p)
                    for fig_k in fig_d:
                        fig_id = os.path.join(path, fig_k + 'range_{}_{}-{}-fit_{}-step_{}-conf_{}'.format(before_range, after_range, fig_k, p['fit'], p['delay_step_size'], p['confidence']))
                        saving_utils.save_plotly_fig(fig_d[fig_k], fig_id)
                        saving_utils.save_csv_dict(bin_stats, path=fig_id + '.csv', key_order=['x', 'mean', 'std', 'confidence_95'])


                    if p['delay_step_size'] == 10:
                        data_csv_path = os.path.join(path, 'range_{}_{}-step_{}-all.csv'.format(before_range, after_range, p['delay_step_size']))
                        DataFrame(bin_stats['y_all'], columns=bin_stats['x']).to_csv(data_csv_path, index=False)
                    '''
        """

    def plot_all_mult(self, AA_l):
        output_experiment_path_comparison, combine_str, day_l_s, AA_l_s = self.setup_comparison_vars(AA_l)
        output_experiment_path_comparison = '/Users/achilleasgeorgiou/Desktop/data_output/astro_axon/180912_003/180912_003_day_0-181012_002_day_0-181022_003_day_0/'
        self.setup_plot_folders_all(output_experiment_path_comparison)

        '''
        print('Saving results of ratios running, rest, stick-running, stick-rest of each axon-astrocyte in csv...')
        astro_ratios_np = np.zeros([len(AA_l), 6])
        r = [AA.id for AA in AA_l]
        c = ['running', 'rest', 'stick_run_ind_15', 'stick_rest', 'total_time_s', 'total_time_m']
        c_n = ['running', 'rest', 'stick_run', 'stick_rest', 'total_time(s)', 'total_time(m)']
        for i, AA in enumerate(AA_l):
            num_frames = len(AA.indices_d['default'])
            num_seconds = num_frames / AA.fr
            num_minutes = general_utils.truncate(num_seconds / 60.0, 2)
            num_seconds = general_utils.truncate(num_seconds, 2)
            for j, k in enumerate(c):
                if j == 4:
                    astro_ratios_np[i, j] = num_seconds
                    continue
                if j == 5:
                    astro_ratios_np[i, j] = num_minutes
                    continue
                if k not in AA.indices_d:
                    print('Not exist', k, AA.id)
                    astro_ratios_np[i, j] = 0
                    continue
                astro_ratios_np[i, j] = general_utils.truncate(len(AA.indices_d[k]) / num_frames, 3)

        behaviour_ratios_csv_path = os.path.join(output_experiment_path_comparison, 'data', 'behaviour_ratios', 'ratios.csv')
        DataFrame(astro_ratios_np, columns=c, index=r).to_csv(behaviour_ratios_csv_path)
        '''
        """
        #Get correlation figures - axon-astro_1, axon-astro_2. What is max correlation, what is the delay
        #etc. after respective axon fires with axon
        for time_range in [[0, 10], [0, 15], [0, 20], [0, 30], [0, 50]]:
            for inverse in [True, False]:
                for bh in ['default', 'stick', 'rest', 'running']:
                    figs, stats, correlated_figs = self.get_main_correlation_figs(AA_l_s, bh=bh, time_range=time_range, inverse=inverse)

                    correlations_folder = 'correlation_results_inverse' if inverse else 'correlation_results'
                    print('Correlations folder: {} inverse: {}'.format(correlations_folder, inverse))

                    for k in figs.keys():
                        path =  os.path.join(output_experiment_path_comparison, 'plots', correlations_folder, 'd_range({}-{})_{}_{}'.format(time_range[0], time_range[1], k, bh))
                        saving_utils.save_plotly_fig(figs[k], path)
                        stats_d = stats[k]
                        #{'x' : x, 'data' : y, 'conf_95' : conf_l, 'std' : std_l, 'mean' : mean_l, 'names' : x}
                        saving_utils.save_csv_dict(stats_d, path + '-stats.csv', key_order=['x', 'mean', 'conf_95', 'std'])
                        print(stats_d['data'])
                        saving_utils.save_csv_dict({stats_d['x'][i] : y_l for i, y_l in enumerate(stats_d['data'])}, path + '-data.csv', key_order=stats_d['x'])

                    #saving_utils.save_plotly_fig(figs['max_corr'], os.path.join(output_experiment_path_comparison, 'plots', correlations_folder, 'd_range({}-{})_max_corr_{}'.format(time_range[0], time_range[1], bh)))
                    #saving_utils.save_plotly_fig(figs['corr_delays'], os.path.join(output_experiment_path_comparison, 'plots', correlations_folder, 'd_range({}-{})_corr_delays_{}'.format(time_range[0], time_range[1], bh)))
                    #saving_utils.save_plotly_fig(figs['event_delays'], os.path.join(output_experiment_path_comparison, 'plots', correlations_folder, 'd_range({}-{})_event_delays_{}'.format(time_range[0], time_range[1], bh)))

                    #TODO fix plots to be in correct time relative to each other
                    for result_i in correlated_figs.keys():
                        for k in correlated_figs[result_i].keys():
                            for sample_num, fig in enumerate(correlated_figs[result_i][k]):
                                if fig is not None:
                                    try:
                                        samples_folder = 'correlation_signal_samples_inverse' if inverse else 'correlation_signal_samples'
                                        os.makedirs(os.path.join(output_experiment_path_comparison, 'plots', samples_folder, 'd_range({}-{})_{}_{}_sample-{}_{}'.format(time_range[0], time_range[1], result_i, k, sample_num, bh)))
                                    except:
                                        pass
                                    saving_utils.save_plotly_fig(fig, os.path.join(output_experiment_path_comparison, 'plots', samples_folder, 'd_range({}-{})_{}_{}_{}-sample-{}'.format(time_range[0], time_range[1], result_i, k, bh, sample_num)))
        """

        '''
        signal_delays_path = os.path.join(output_experiment_path_comparison, 'plots' , 'signal_delays_axon')
        print('Plotting signal axon delays waterfall')
        fig_delays_waterfall_d, fig_delays_waterfall_interpolate_d = self.get_waterfall_delays_axons_far_plot(AA_l)

        for fig_k in fig_delays_waterfall_d.keys():
            print('FIG K', fig_k)
            saving_utils.save_plotly_fig(fig_delays_waterfall_d[fig_k], os.path.join(signal_delays_path, fig_k + '-delays_waterfall'))
            saving_utils.save_plotly_fig(fig_delays_waterfall_interpolate_d[fig_k], os.path.join(signal_delays_path, fig_k + '-delays_waterfall_interpolate'))


        signal_axon_border_delays_path = os.path.join(output_experiment_path_comparison, 'plots' , 'signal_delays_axon_border')
        print('Plotting signal axon border only delays waterfall')
        fig_axon_border_delays_waterfall_d, fig_axon_border_delays_waterfall_interpolate_d = self.get_axon_only_bound_waterfall_latencies_plot(AA_l)


        for fig_k in fig_axon_border_delays_waterfall_d.keys():
            print('FIG K', fig_k)
            saving_utils.save_plotly_fig(fig_axon_border_delays_waterfall_d[fig_k], os.path.join(signal_axon_border_delays_path, fig_k + '-delays_waterfall'))
            saving_utils.save_plotly_fig(fig_axon_border_delays_waterfall_interpolate_d[fig_k], os.path.join(signal_axon_border_delays_path, fig_k + '-delays_waterfall_interpolate'))
        '''

        '''
        axon_only_proportion_delay_path = os.path.join(output_experiment_path_comparison, 'plots', 'prop_signal_delays_axon_border')
        fig_axon_only_prop_d = self.get_axon_only_proportion_delays_plot(AA_l, min_delay=-20, max_delay=50)

        for fig_k in fig_axon_only_prop_d.keys():
            saving_utils.save_plotly_fig(fig_axon_only_prop_d[fig_k], os.path.join(axon_only_proportion_delay_path, fig_k))


        axon_only_proportion_delay_avg_path = os.path.join(output_experiment_path_comparison, 'plots', 'prop_signal_delays_axon_border_avg')
        fig_axon_only_prop_d = self.get_axon_only_proportion_delays_plot(AA_l, min_delay=-20, max_delay=50, avg=True)

        for fig_k in fig_axon_only_prop_d.keys():
            saving_utils.save_plotly_fig(fig_axon_only_prop_d[fig_k], os.path.join(axon_only_proportion_delay_avg_path, fig_k))

        axon_only_proportion_delay_gauss_path = os.path.join(output_experiment_path_comparison, 'plots', 'prop_signal_delays_axon_border_gauss_filter')
        fig_axon_only_prop_d = self.get_axon_only_proportion_delays_plot(AA_l, min_delay=-20, max_delay=50, gauss_f=True)

        for fig_k in fig_axon_only_prop_d.keys():
            saving_utils.save_plotly_fig(fig_axon_only_prop_d[fig_k], os.path.join(axon_only_proportion_delay_gauss_path, fig_k))
        '''

        '''
        path = os.path.join(output_experiment_path_comparison, 'plots', 'waterfall_together')
        fig_waterfall_all_d, fig_waterfall_all_interp_d = self.get_waterfall_delays_together_axons_far_plot(AA_l)

        for k in fig_waterfall_all_d.keys():
            saving_utils.save_plotly_fig(fig_waterfall_all_d[k], os.path.join(path, '{}-waterfall'.format(k)))
            saving_utils.save_plotly_fig(fig_waterfall_all_interp_d[k], os.path.join(path, '{}-waterfall_interp'.format(k)))
        '''


        print('---TRANSITION PROPORTION DELAYS PLOT ALL---')
        delay_ranges_pairs = [ [3*AA_l[0].fr, 6*AA_l[0].fr],
                               #[1*AA_l[0].fr, 1*AA_l[0].fr],
                               [2*AA_l[0].fr, 4*AA_l[0].fr]]

        delay_ranges_pairs = [[int(v[0]), int(v[1])] for v in delay_ranges_pairs]

        print('Alt Proportion plots...')
        #for setting in ['axon', 'astro']:
        for setting in ['axon', 'astro']:
            for delay_ranges_pair in delay_ranges_pairs:
                before_range, after_range = delay_ranges_pair
                for p in [{'fit' : True, 'delay_step_size' : 1, 'confidence' : True, 'setting' : setting},
                          #{'fit' : True, 'delay_step_size' : 5, 'confidence' : True},
                          #{'fit' : True, 'delay_step_size' : 10, 'confidence': True, 'setting' : setting}
                          ]:

                    ################################################
                    ##############Proportion plots##################
                    ################################################

                    print('EXTRA PARS', p, p.keys())

                    print('rest to run prop')
                    path = os.path.join(output_experiment_path_comparison, 'plots', 'rest_to_run_{}_proportions'.format(setting))
                    fig_d, bin_stats = self.get_axon_transition_proportion_delays_plot_all(AA_l, before_bh='rest_semi_exact', inds_bh='running_exact_start', after_bh='running_semi_exact',
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
                    path = os.path.join(output_experiment_path_comparison, 'plots', 'run_to_rest_{}_proportions'.format(setting))
                    fig_d, bin_stats = self.get_axon_transition_proportion_delays_plot_all(AA_l, before_bh='running_semi_exact', inds_bh='rest_start', after_bh='rest_semi_exact',
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
                    path = os.path.join(output_experiment_path_comparison, 'plots', 'run_stick_run_{}_proportions'.format(setting))
                    fig_d, bin_stats = self.get_axon_transition_proportion_delays_plot_all(AA_l, before_bh='running_semi_exact', inds_bh='stick_exact_start', after_bh='running_semi_exact',
                                                                                    before_range=before_range, after_range=after_range,
                                                                                    **p)
                    for fig_k in fig_d:
                        fig_id = os.path.join(path, fig_k + 'range_{}_{}-{}-fit_{}-step_{}-conf_{}'.format(before_range, after_range, fig_k, p['fit'], p['delay_step_size'], p['confidence']))
                        saving_utils.save_plotly_fig(fig_d[fig_k], fig_id)
                        saving_utils.save_csv_dict(bin_stats, path=fig_id + '.csv', key_order=['x', 'mean', 'std', 'confidence_95'])


                    if p['delay_step_size'] == 10:
                        data_csv_path = os.path.join(path, 'range_{}_{}-step_{}-all.csv'.format(before_range, after_range, p['delay_step_size']))
                        DataFrame(bin_stats['y_all'], columns=bin_stats['x']).to_csv(data_csv_path, index=False)



                    print('rest to run prop, STICK FILTERED')
                    path = os.path.join(output_experiment_path_comparison, 'plots', 'rest_to_run_STICK_FILT_{}_proportions'.format(setting))
                    fig_d, bin_stats = self.get_axon_transition_proportion_delays_STICK_FILTER_plot_all(AA_l, before_bh='rest_semi_exact', inds_bh='running_exact_start', after_bh='running_semi_exact',
                                                                                    before_range=before_range, after_range=after_range,
                                                                                    **p)
                    for fig_k in fig_d:
                        fig_id = os.path.join(path, fig_k + 'range_{}_{}-{}-fit_{}-step_{}-conf_{}'.format(before_range, after_range, fig_k, p['fit'], p['delay_step_size'], p['confidence']))
                        saving_utils.save_plotly_fig(fig_d[fig_k], fig_id)
                        saving_utils.save_csv_dict(bin_stats, path=fig_id + '.csv', key_order=['x', 'mean', 'std', 'confidence_95'])


                    if p['delay_step_size'] == 10:
                        data_csv_path = os.path.join(path, 'range_{}_{}-step_{}-all.csv'.format(before_range, after_range, p['delay_step_size']))
                        DataFrame(bin_stats['y_all'], columns=bin_stats['x']).to_csv(data_csv_path, index=False)

                    print('run to rest prop, STICK FILTERED')
                    path = os.path.join(output_experiment_path_comparison, 'plots', 'run_to_rest_STICK_FILT_{}_proportions'.format(setting))
                    fig_d, bin_stats = self.get_axon_transition_proportion_delays_STICK_FILTER_plot_all(AA_l, before_bh='running_semi_exact', inds_bh='rest_start', after_bh='rest_semi_exact',
                                                                                    before_range=before_range, after_range=after_range,
                                                                                    **p)
                    for fig_k in fig_d:
                        fig_id = os.path.join(path, fig_k + 'range_{}_{}-{}-fit_{}-step_{}-conf_{}'.format(before_range, after_range, fig_k, p['fit'], p['delay_step_size'], p['confidence']))
                        saving_utils.save_plotly_fig(fig_d[fig_k], fig_id)
                        saving_utils.save_csv_dict(bin_stats, path=fig_id + '.csv', key_order=['x', 'mean', 'std', 'confidence_95'])

                    if p['delay_step_size'] == 10:
                        data_csv_path = os.path.join(path, 'range_{}_{}-step_{}-all.csv'.format(before_range, after_range, p['delay_step_size']))
                        DataFrame(bin_stats['y_all'], columns=bin_stats['x']).to_csv(data_csv_path, index=False)


                    print('run stick hit run prop, STICK FILTERED')
                    path = os.path.join(output_experiment_path_comparison, 'plots', 'run_stick_run_STICK_FILT_{}_proportions'.format(setting))
                    fig_d, bin_stats = self.get_axon_transition_proportion_delays_STICK_FILTER_plot_all(AA_l, before_bh='running_semi_exact', inds_bh='stick_exact_start', after_bh='running_semi_exact',
                                                                                    before_range=before_range, after_range=after_range,
                                                                                    **p)
                    for fig_k in fig_d:
                        fig_id = os.path.join(path, fig_k + 'range_{}_{}-{}-fit_{}-step_{}-conf_{}'.format(before_range, after_range, fig_k, p['fit'], p['delay_step_size'], p['confidence']))
                        saving_utils.save_plotly_fig(fig_d[fig_k], fig_id)
                        saving_utils.save_csv_dict(bin_stats, path=fig_id + '.csv', key_order=['x', 'mean', 'std', 'confidence_95'])


                    if p['delay_step_size'] == 10:
                        data_csv_path = os.path.join(path, 'range_{}_{}-step_{}-all.csv'.format(before_range, after_range, p['delay_step_size']))
                        DataFrame(bin_stats['y_all'], columns=bin_stats['x']).to_csv(data_csv_path, index=False)



        print('Proportions ALL axon and astro')
        print('Alternative run-rest/rest-run averaging individual lines')
        delay_ranges_pairs = [ [3*AA_l[0].fr, 6*AA_l[0].fr],
                               [1*AA_l[0].fr, 1*AA_l[0].fr],
                               [2*AA_l[0].fr, 4*AA_l[0].fr]]
        delay_ranges_pairs = [[int(v[0]), int(v[1])] for v in delay_ranges_pairs]

        bh_measure_l = ['speed']
        bh_measure_path_l = ['speed']
        bh_measure_y_titles = ['Speed (cm/s)']
        print('BEHAVIOUR BH MEASURE PLOTS')
        for delay_ranges_pair in delay_ranges_pairs:
                before_range, after_range = delay_ranges_pair
                for p in [{'fit' : True, 'delay_step_size' : 1, 'confidence' : True},
                          #{'fit' : True, 'delay_step_size' : 5, 'confidence' : True},
                          #{'fit' : True, 'delay_step_size' : 10, 'confidence': True}
                          ]:
                    ################################################
                    ##############Behaviour measure plots###########
                    ################################################

                    for m_i, bh_measure in enumerate(bh_measure_l):

                        print('BH measure {} rest-run'.format(bh_measure))
                        path = os.path.join(output_experiment_path_comparison, 'plots', 'rest_to_run_{}'.format(bh_measure_path_l[m_i]))
                        fig_d, bin_stats = self.get_transition_bh_values_plot_all_alt(AA_l,
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

                        path = os.path.join(output_experiment_path_comparison, 'plots', 'run_to_rest_{}'.format(bh_measure_path_l[m_i]))
                        fig_d, bin_stats = self.get_transition_bh_values_plot_all_alt(AA_l,
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

                        path = os.path.join(output_experiment_path_comparison, 'plots', 'run_stick_run_{}'.format(bh_measure_path_l[m_i]))
                        fig_d, bin_stats = self.get_transition_bh_values_plot_all_alt(AA_l,
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

    def get_delays_after_count_plots(self, AA):
        delays_after_count_plots = {}

        for bk in AA.indices_d.keys():
            delays_after_count_plots[bk] = {}
            for axon_str in AA.axon_strs:
                x = []
                y = []
                strs = []
                for landmark_id in AA.num_events[bk][axon_str].keys():
                    total_events = len(AA.times_contained[bk][AA.compare_tuples[axon_str][landmark_id]])
                    events_after_all = general_utils.truncate(AA.num_events_norm[bk][axon_str][landmark_id], 2)

                    x.append(landmark_id)
                    y.append(events_after_all)
                    strs.append(total_events)

                arg_s_i = np.argsort(y)
                x_s = np.array(x)[arg_s_i]
                y_s = np.array(y)[arg_s_i]
                strs_s = np.array(strs)[arg_s_i]

                fig = plotly_utils.plot_bar(x=x_s, y=y_s, text_values=strs_s, text_size=20, title='Ratio of events after {} in landmark within (delay {}-{}s)'.format(axon_str, AA.delay_min, general_utils.truncate(AA.delay_max/AA.fr, 1)), x_title='Landmark id', y_title='(%) of total events in landmark')
                self.apply_fun_axis_fig(fig, lambda x : x * 100, axis='y')
                delays_after_count_plots[bk][axon_str] = fig

        return delays_after_count_plots

    def get_random_landmark_event_df_plots(AA):
        for bk in AA.indices_d.keys():
            for axon_str in axon_strs:
                AA.events_contained[bk][axon_str]

    def events_after_count_csvs(self, AA, csv_path):
        # Save to .csv files
        for bk in AA.indices_d.keys():
            print('................')
            print(bk)
            for axon_str in AA.axon_strs:
                with open(os.path.join(csv_path, bk + '_' + axon_str + '.csv'), mode='w') as csv_file:
                    writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    writer.writerow(['Landmark', 'Total events', 'Events after', 'Events after/Events all|', 'Events after/Events axon'])
                    for landmark_id in AA.num_events[bk][axon_str].keys():
                        if AA.num_events[bk][axon_str]['axon'] != 0:
                            total_events = len(AA.times_contained[bk][AA.compare_tuples[axon_str][landmark_id]])
                            events_after = AA.num_events[bk][axon_str][landmark_id]
                            events_after_all = general_utils.truncate(AA.num_events_norm[bk][axon_str][landmark_id], 2)
                            events_after_axon = general_utils.truncate(AA.num_events[bk][axon_str][landmark_id]/AA.num_events[bk][axon_str]['axon'], 2)
                            writer.writerow([landmark_id, total_events, events_after, events_after_all, events_after_axon])

    def delays_per_landmark_csvs(self, AA, csv_path):
        for bk in AA.indices_d.keys():
            for axon_str in AA.axon_strs:
                with open(os.path.join(csv_path, bk + '_'  + axon_str + '.csv'), mode='w') as csv_file:
                    writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    delay_landmark_ids, delay_means, delay_stds = AA.get_values_heatmap_from_landmarks(AA.ddr_heatmaps[bk][axon_str],
                                                                                                 AA.ddr_heatmaps_count[bk][axon_str],
                                                                                                 AA.landmark_locs)
                    writer.writerow(['Landmark', 'Mean delay', 'Std delay'])
                    for i in range(len(delay_landmark_ids)):
                        try:
                            writer.writerow([delay_landmark_ids[i], general_utils.truncate(delay_means[i],2), general_utils.truncate(delay_stds[i], 2)])
                        except ValueError:
                            writer.writerow([delay_landmark_ids[i], np.nan, np.nan])

    def get_behaviour_basic_plots(self, AA):
        fig_stick = plotly_utils.plot_scatter_fmt(x=np.arange(len(AA.stick_bin)), y=AA.stick_bin, astype='int', straight_lines_only=True, title='Stick', x_title='Frame', y_title='Off whisker/On whisker')
        fig_speed = plotly_utils.plot_scatter_fmt(x=np.arange(len(AA.speed_bin)), y=AA.speed_bin, astype='int', straight_lines_only=True, title='Speed', x_title='Frame', y_title='Rest/Running')
        fig_pupil = plotly_utils.plot_scatter_fmt(x=np.arange(len(AA.pupil_values)), y=AA.pupil_values, astype='float', straight_lines_only=True, title='Pupil', x_title='Frame', y_title='Pupil value')
        return fig_stick, fig_speed, fig_pupil

    def get_behaviour_contour_plots(self, AA):
        fig_heatmap_grids = {}
        fig_heatmap_dff_grids = {}

        #fig_heatmap_dff_grids
        for k in AA.event_subsets.keys():
            fig_heatmap_grids[k] = plotly_utils.plot_contour(AA.event_grids[k], title=k + '_event grid', height=600, width=800)

        for k in AA.event_subsets.keys():
            fig_heatmap_dff_grids[k] = plotly_utils.plot_contour(AA.event_grids_dff[k], title=k+'_event grid dff', height=600, width=800)

        return fig_heatmap_grids, fig_heatmap_dff_grids

    def get_behaviour_activity_plot(self, AA):
        activity_ratio_k = np.array(['default', 'running', 'rest', 'stick'])
        activity_ratio_l = np.array([AA.activity_ratios[k] for k in activity_ratio_k])
        text_values =  np.array(['Frames: ' + str(len(AA.indices_d[k])) for k in activity_ratio_k])
        activity_i = np.argsort(activity_ratio_l)

        activity_ratio_k_s = activity_ratio_k[activity_i]
        activity_ratio_l_s = activity_ratio_l[activity_i]
        text_values_s = text_values[activity_i]

        activity_ratio_k_s[np.where(activity_ratio_k_s == 'default')] = 'all'
        fig = plotly_utils.plot_bar(x=activity_ratio_k_s, y=activity_ratio_l_s, text_values=text_values_s, text_size=20, title='Activity ratio (events per voxel)', x_title='Behaviour', y_title='Events per voxel (%)')
        self.apply_fun_axis_fig(fig, lambda x : x * 100, axis='y',)
        #fig['layout'].update(yaxis=dict(ticksuffix= "%"))
        return fig

    def get_ddr_heatmap_figs(self, AA):
        figs_drr_heatmaps = {}
        for bk in AA.indices_d.keys():
            figs_drr_heatmaps[bk] = {}
            for axon_k in AA.axon_strs:
                figs_drr_heatmaps[bk][axon_k] = plotly_utils.plot_contour(AA.ddr_heatmaps[bk][axon_k], title='ddr plot ' + bk + ' ' + axon_k, height=600, width=800)
        return figs_drr_heatmaps

    def get_landmark_border_figs(self, AA):
        figs_landmark_borders = {}
        figs_landmark_lines = {}
        for lk in AA.landmark_borders.keys():
            figs_landmark_borders[lk] = plotly_utils.plot_contour(AA.landmark_borders[lk], title='Borders ' + lk, height=600, width=800)
            figs_landmark_lines[lk] = plotly_utils.plot_contour(AA.landmark_line_borders[lk], title = 'Line borders ' + lk, height=600, width=800)
        return figs_landmark_borders, figs_landmark_lines

    def apply_fun_axis_fig(self, fig, fun, axis='x'):
        if axis == 'x':
            for i in range(len(fig['data'])):
                fig['data'][i]['x'] = fun(np.array(fig['data'][i]['x']))
        elif axis == 'y':
            for i in range(len(fig['data'])):
                fig['data'][i]['y'] = fun(np.array(fig['data'][i]['y']))
        else:
            print('Invalid axis value in apply_fun_axis_fig. Pass \'x\' or \'y\'')

    def get_signal_fig(self, AA, event_i, add_range=100):
        divisor = AA.res_d['dff_only'].shape[1]
        t_begin = int(AA.res_d['tBegin'][event_i] % divisor)
        t_end = int(AA.res_d['tEnd'][event_i] % divisor)
        if (t_begin - add_range) > (t_end + add_range):
            print('Tbegin > tEnd')
            return None

        if t_begin-add_range <= 0:
            print('Tbegin - add_range < 0')
            return None

        y = AA.res_d['dff_only'][event_i][t_begin-add_range:t_end+add_range]
        x = np.arange(len(y))

        fig = plotly_utils.plot_scatter_signal(x=x, y=y, begin_i=add_range, end_i=t_end - t_begin + add_range, mode='lines', title='Signal', x_title='', y_title='')
        return fig

    #TODO
    def get_signal_fig_mult(self, AA, event_id_l, add_range=100):
        divisor = AA.res_d['dff_only'].shape[1]

        min_start = AA.res_d['tBegin'][event_id_l[0]]
        max_end = AA.res_d['tEnd'][event_id_l[0]]

        for event_id_i in event_id_l:
            curr_min = AA.res_d['tBegin'][event_id_i]
            curr_max = AA.res_d['tEnd'][event_id_i]

            print(curr_min, min_start)
            if curr_min < min_start:
                min_start = curr_min
            if curr_max > max_end:
                max_end = curr_max

        if max_end - min_start > divisor:
            print('Min start: {}, max start: {}'.format(min_start, max_start))
            return None
        if max_end % divisor < min_start % divisor:
            print('Max start div < min start div')
            return None
        if min_start-add_range <= 0:
            print('Tbegin - add_range < 0')
            return None
        if max_end+add_range % divisor < max_end:
            print('Max start + add range passes div')
            return None

        t_begin_l = [int(AA.res_d['tBegin'][e_i]) for e_i in event_id_l]
        t_end_l = [int(AA.res_d['tEnd'][e_i]) for e_i in event_id_l]

        t_begin_div_l = [(t_begin % divisor) for t_begin in t_begin_l]
        t_end_div_l = [(t_end % divisor) for t_end in t_end_l]

        min_start_div = int(min_start % divisor)
        max_end_div = int(max_end % divisor)

        print(min_start_div)
        print(max_end_div)

        y_l = [AA.res_d['dff_only'][e_i][min_start_div-add_range:max_end_div+add_range] for e_i in event_id_l]
        x = np.arange(len(y_l[0]))

        t_begin_l_x = [add_range + t_begin_div - min_start_div for t_begin_div in t_begin_div_l]
        t_end_l_x = [add_range + (max_end_div - min_start_div) - (max_end_div - t_end_div) for t_end_div in t_end_div_l]

        fig = plotly_utils.plot_scatter_signal_mult(x=x, y_l=y_l, begin_l=t_begin_l_x, end_l=t_end_l_x, mode='lines', title='Signal', x_title='', y_title='')
        return fig

    def get_signal_figs_samples(self, AA, sample_num=20):
        event_sample_inds = np.random.choice(len(AA.res_d['tBegin']), sample_num, replace=False)
        figs = []
        for event_i in event_sample_inds:
            figs.append(self.get_signal_fig(AA, event_i))
        return figs

    def get_signal_landmark_figs_samples(self, AA, sample_num=10):
        figs = {}
        for bk in AA.events_contained.keys():
            print('BK', bk)
            figs[bk] = {}
            for lk in AA.events_contained[bk]:
                print('BK LK', bk, lk)
                sample_num_x = min(len(AA.events_contained[bk][lk]), sample_num)
                event_sample_inds = np.random.choice(AA.events_contained[bk][lk], sample_num_x, replace=False)
                figs[bk][lk] = []

                for event_i in event_sample_inds:
                    figs[bk][lk].append(self.get_signal_fig(AA, event_i))
        print('Done, landmark figs, returning')
        return figs

    #####################################################
    def get_main_correlation_figs(self, AA_l, bh='default', time_range=[0, 15], inverse=False):
        all_results_l = []
        for AA in AA_l:
            all_results_l.append(self.get_axon_astro_correlations(AA, bh=bh, time_range=time_range, inverse=inverse))
        results_d = self.merge_results(all_results_l)

        figs = {}
        stats = {}
        keys = ['axon_far', 'astro_close_1', 'astro_close_2', 'astro_far']
        fig, stats_d = plotly_utils.plot_point_box_revised(keys, [results_d['max_corr'][k] for k in keys], title='{} max corr'.format(bh), with_stats=True)
        figs['max_corr'] = fig
        stats['max_corr'] = stats_d
        fig, stats_d = plotly_utils.plot_point_box_revised(keys, [(np.array(results_d['corr_delays'][k]) / AA.fr) for k in keys], title='{} corr delay'.format(bh), y_title='Delay (s)', with_stats=True)
        figs['corr_delays'] = fig
        stats['corr_delays'] = stats_d
        fig, stats_d = plotly_utils.plot_point_box_revised(keys, [(np.array(results_d['event_delays'][k]) / AA.fr) for k in keys], title='{} event delay'.format(bh), y_title='Delay (s)', with_stats=True)
        figs['event_delays'] = fig
        stats['event_delays'] = stats_d
        #Generate random sample plots (correlated)
        correlated_figs = {}

        for result_i, result in enumerate(all_results_l):
            result_k = AA_l[result_i].print_id
            correlated_figs[result_k] = {} #TODO

            for k in ['astro_close_1', 'astro_close_2', 'astro_far', 'axon_far']:
                correlated_figs[result_k][k] = []
                low_delay_pair_inds = np.argsort(result['event_delays_raw'][k][result_i])
                for i in range(min(self.n_signal_samples, len(low_delay_pair_inds))):
                    ev0, ev1 = result['corr_tuples'][k][result_i][low_delay_pair_inds[i]]
                    #fig_0 = self.get_signal_fig(AA_l[result_i], ev0)
                    #fig_1 = self.get_signal_fig(AA_l[result_i], ev1)

                    fig = self.get_signal_fig_mult(AA_l[result_i], [ev0, ev1])

                    correlated_figs[result_k][k].append(fig)
        return figs, stats, correlated_figs

    def get_axon_astro_correlations(self, AA, bh='default', time_range=[0,15], inverse=False):
        # ['astro_3_close_1', 'axon_2_far', 'astro_3_close_2', 'axon_3_far', 'all', 'astro_far_1', 'axon_1_far', 'astro_far_2', 'astro_far_3', 'cell_bound', 'axons', 'axon_1', 'axon_2', 'axon_3', 'astro_2_close_2', 'astro_2_close_1', 'astro_1_close_1', 'astro_1_close_2']
        # (axon - astro_close_1)
        # (axon - astro_close_2)
        # (axon - axon_far)
        # (axon - astro_far_1)
        # (axon - astro_far_2)

        #inverse: to get astro axon correlations, that is axon event after astro events
        results_d = {'corr_delays' : {}, 'event_delays': {}, 'max_corr' : {}, 'corr_tuples' : {},
                     'corr_delays_raw' : {}, 'event_delays_raw' : {}, 'max_corr_raw' : {}}
        for k in results_d.keys():
            results_d[k] = {'axon_self' : [], 'axon_far': [], 'astro_close_1' : [], 'astro_close_2' : [], 'astro_far' : []}

        def add_to_results(results_d, result, result_id, operation='mean'):
            for k in result.keys():
                if k in ['corr_tuples']:
                    results_d[k][result_id].append(result[k])
                    continue
                if k in ['corr_delays', 'event_delays', 'max_corr']:
                    results_d[k + '_raw'][result_id].append(result[k])
                if operation == 'mean':
                    results_d[k][result_id].append(np.mean(result[k]))
                else:
                    print('Only operation : (mean) supported')

        for axon_num in range(1, AA.num_axons+1):
            ev_axon = AA.events_contained[bh]['axon_{}'.format(axon_num)]
            ev_axon_far = AA.events_contained[bh]['axon_{}_far'.format(axon_num)]
            ev_close_1 = AA.events_contained[bh]['astro_{}_close_1'.format(axon_num)]
            ev_close_2 = AA.events_contained[bh]['astro_{}_close_2'.format(axon_num)]

            if inverse:
                add_to_results(results_d, self.get_event_correlations(AA, ev_axon, ev_axon, time_range=time_range), 'axon_self')
                add_to_results(results_d, self.get_event_correlations(AA, ev_close_1, ev_axon, time_range=time_range), 'astro_close_1')
                add_to_results(results_d, self.get_event_correlations(AA, ev_close_2, ev_axon, time_range=time_range), 'astro_close_2')
                add_to_results(results_d, self.get_event_correlations(AA, ev_axon_far, ev_axon, time_range=time_range), 'axon_far')
            else:
                add_to_results(results_d, self.get_event_correlations(AA, ev_axon, ev_axon, time_range=time_range), 'axon_self')
                add_to_results(results_d, self.get_event_correlations(AA, ev_axon, ev_close_1, time_range=time_range), 'astro_close_1')
                add_to_results(results_d, self.get_event_correlations(AA, ev_axon, ev_close_2, time_range=time_range), 'astro_close_2')
                add_to_results(results_d, self.get_event_correlations(AA, ev_axon, ev_axon_far, time_range=time_range), 'axon_far')

            for astro_far_num in range(1, AA.num_astro_far+1):
                ev_far_i = AA.events_contained[bh]['astro_far_{}'.format(astro_far_num)]

                if inverse:
                    add_to_results(results_d, self.get_event_correlations(AA,ev_far_i, ev_axon, time_range=time_range), 'astro_far')
                else:
                    add_to_results(results_d, self.get_event_correlations(AA, ev_axon, ev_far_i, time_range=time_range), 'astro_far')
        return results_d

    def get_event_correlations(self, AA, ev1, ev2, mode='dff', time_range=[0,15]):
        '''
        Given two event lists, find all pairs of events that are close in time and correlate them,
        recording the delay and correlation value
        '''
        #Event delay is difference in start time between ev1_i, ev2_i
        event_delay_l = []
        #Corr delay is how much the ev2_i signal must be moved back to match max correlation with ev1_i signal
        corr_delay_l = []
        #Normalized max correlation between ev1_i and ev2_i
        max_corr_l = []

        #Find event pairs that are close in time (0:50) frames
        corr_tuple_l = self.get_time_range_tuple_events(AA, ev1, ev2, time_range=time_range)

        #If they are in different video segments (rare occurence) we do not have df/f values for both signals, so we ignore this
        corr_tuple_l_f = [corr_tuple for corr_tuple in corr_tuple_l if
                            AA.res_d['event_i_video_segment'][corr_tuple[0]] == AA.res_d['event_i_video_segment'][corr_tuple[1]]]

        ###print('len before:', len(corr_tuple_l), 'len after:', len(corr_tuple_l_f))

        #for each event pair
            #correlate ev1[start_ev1:end_ev2] with ev2[start_ev2:end_ev2]
            #record delay and normalized correlation value

        print('CORR TUPLE L F', corr_tuple_l_f)

        for corr_tuple in corr_tuple_l_f:
            signal_1_start = AA.res_d['tBegin'][corr_tuple[0]]
            signal_2_start = AA.res_d['tBegin'][corr_tuple[1]]
            signal_delay = signal_2_start - signal_1_start

            signal_1_full = AA.res_d['dff_only'][corr_tuple[0]]
            signal_2_full = AA.res_d['dff_only'][corr_tuple[1]]

            signal_1_range = self.event_i_time_segment(AA, corr_tuple[0])
            signal_2_range = self.event_i_time_segment(AA, corr_tuple[1])

            #if signal range spills to the next segment, same thing
            if (signal_1_range is None) or (signal_2_range is None):
                print('Does this happen??')
                continue

            #Take ev1[start_ev1:end_ev2]
            signal_1 = signal_1_full[np.arange(signal_1_range[0], signal_2_range[1])]
            #Take ev2[start_ev2:end_ev2]
            signal_2 = signal_2_full[np.arange(signal_2_range[0], signal_2_range[1])]

            max_corr, max_corr_i = correlation_utils.get_max_cross_correlation(signal_1, signal_2, normalize=True)

            event_delay_l.append(signal_delay)
            corr_delay_l.append(max_corr_i)
            max_corr_l.append(max_corr)

        return {'corr_delays' : corr_delay_l, 'event_delays': event_delay_l, 'max_corr' : max_corr_l, 'corr_tuples' : corr_tuple_l_f}

    def get_time_range_tuple_events(self, AA, ev1, ev2, time_range=(0,50)):
        '''
        Get all pairs of events from (ev1, ev2) that are within time range specified starting from ev1.
        e.g. ev1_i starts at 110, ev2_j starts at 130, then 20 delay in time range, so (ev1_i, ev2_j) event added as tuple
        '''
        ev1_start = AA.res_d['tBegin'][ev1]
        ev1_end = AA.res_d['tEnd'][ev1]
        ev2_start = AA.res_d['tBegin'][ev2]
        ev2_end = AA.res_d['tEnd'][ev2]

        corr_tuple_l = []

        for ev2_i in range(len(ev2)):
            print('TIME RNAGE', time_range[0], time_range[1])
            diff_arr = ev2_start[ev2_i] - ev1_start
            valid_ev1_i_l = np.where((diff_arr >= time_range[0]) & (diff_arr <= time_range[1]))[0]

            #print('VALID', valid_ev1_i_l, len(valid_ev1_i_l))
            #If no event within period of 0:start_delay_max frames do nothing
            if len(valid_ev1_i_l) == 0:
                ###print('Continuing')
                continue
            #Otherwize, add to list correlation tuple (ev1, ev2)
            else:
                for valid_ev1_i in valid_ev1_i_l:
                    corr_tuple_l.append((ev1[valid_ev1_i], ev2[ev2_i]))

                    ###print('Appended events: {} {} with times {} {}'.format(ev1[valid_ev1_i], ev2[ev2_i], ev1_start[valid_ev1_i], ev2_start[ev2_i]))
        return corr_tuple_l

    def event_i_time_segment(self, A, ev, add_start=0, add_end=0):
        '''
        Given event i, return begin time and end of corresponding video segment (e.g. videos here are 2726 frame segments merged together)
        May use sub, add to add/subtract time frames from
        '''
        divisor = A.res_d['dff_only'].shape[1]
        t_begin = int((A.res_d['tBegin'][ev]-add_start) % divisor)
        t_end = int((A.res_d['tEnd'][ev]+add_end) % divisor)
        if t_begin > t_end:
            print('Tbegin > tEnd')
            return None, None
        return (t_begin, t_end)

    def merge_results(self, all_results_l):
        results_d = {}
        for results_d_i in all_results_l:
            for k1 in results_d_i.keys():
                if k1 not in results_d:
                    results_d[k1] = {}
                for k2 in results_d_i[k1].keys():
                    if k2 not in results_d[k1]:
                        results_d[k1][k2] = []
                    results_d[k1][k2].extend(results_d_i[k1][k2])
        return results_d

    #Correlation between whisker stimulation (stick) and axon response in astrocyte-axon interaction experiment
    #Whisker correlation to axon response
    def get_plots_whisker_correlations(self, AA, bh='default', delay_max=50):
        results_d = self.get_whisker_correlation(AA, bh=bh, delay_max=delay_max)
        figs = {}

        figs['max_corr'] = plotly_utils.plot_point_box_revised(['Max correlation'], [results_d['max_corr']], title='{} max corr'.format(bh))
        figs['event_delays'] = plotly_utils.plot_point_box_revised(['Event delays'], [results_d['event_delays']], title='{} event delays'.format(bh))
        figs['corr_delays'] = plotly_utils.plot_point_box_revised(['Corr delays'], [results_d['corr_delays']], title='{} corr delays'.format(bh))

        return figs

    def get_whisker_correlation(self, AA, bh='default', delay_max=50):
        #take stick start indices
        stick_indices = AA.indices_d['stick_exact_start']
        #take all events within 0-5 seconds after start indices
        stick_event_pairs = []
        axon_far_events = []
        axon_far_times = []

        #Get events contained
        for axon_str in AA.axon_strs:
            axon_far_events.extend(AA.events_contained[bh][axon_str + '_far'])
            axon_far_times.extend(AA.times_contained[bh][axon_str + '_far'])

        axon_far_start_times = [axon_far_time[0] for axon_far_time in axon_far_times]

        for stick_index in stick_indices:
            delays_temp = axon_far_start_times - stick_index
            delays_temp_inds = np.where((delays_temp <= 50) & (delays_temp >= 0))

            for i in delays_temp_inds[0]:
                stick_event_pairs.append((stick_index, axon_far_events[i]))

        max_corr_l = []
        corr_delay_l = []
        event_delay_l = []

        for stick_event_pair in stick_event_pairs:
            index, event_i = stick_event_pair
            max_corr, max_corr_i = self.stick_event_correlation(AA, index, event_i)

            max_corr_l.append(max_corr)
            corr_delay_l.append(max_corr_i)
            event_delay_l.append(AA.res_d['tBegin'][event_i] - index)

        return {'corr_delays' : corr_delay_l, 'event_delays': event_delay_l, 'max_corr' : max_corr_l}

    def stick_event_correlation(self, AA, index, event_i):
        '''
        Given index of stick and an event_i get correlation result
        '''
        ev_start = AA.res_d['tBegin'][event_i]
        ev_end = AA.res_d['tEnd'][event_i]
        signal_1 = AA.stick_bin[index:ev_end]
        signal_2_range = self.event_i_time_segment(AA, event_i)
        signal_2 = AA.res_d['dff_only'][event_i][np.arange(signal_2_range[0], signal_2_range[1])]
        max_corr, max_corr_i = correlation_utils.get_max_cross_correlation(signal_1, signal_2, normalize=True)

        return max_corr, max_corr_i

    def get_stick_axon_event_frequency(self, AA, delay=30):
        '''
        After stick hits how much more frequent are axon events compared to default
        '''
        #Take events and times contained for each axon_far
        axon_far_events_d = {}
        axon_far_times_d = {}
        events_after_stick = {}
        for axon_str in AA.axon_strs:
            k = axon_str + '_far'
            axon_far_events_d[k] = AA.events_contained['default'][k]
            axon_far_times_d[k] = AA.times_contained['default'][k]
            events_after_stick[k] = []

        #Look at next 30 frames after stick hits and count number of axon events
        for axon_k in axon_far_times_d.keys():
            axon_far_start_times = [axon_far_time[0] for axon_far_time in axon_far_times_d[axon_k]]

            for stick_index in AA.indices_d['stick_exact_start']:
                delays_temp = axon_far_start_times - stick_index
                delays_temp_inds = np.where((delays_temp <= delay) & (delays_temp >= 0))
                for i in delays_temp_inds[0]:
                    events_after_stick[axon_k].append(axon_far_events_d[axon_k][i])

        stick_start_cont = np.zeros([len(AA.indices_d['default'])])
        stick_start_cont[AA.indices_d['stick_exact_start']] = 1
        stick_start_cont = analysis_pp.get_behaviour_indices(stick_start_cont, 1, ind_past=0, ind_future=delay, complementary=False)
        #How much stick compared to default time?
        print('Ratio (stick/all frames): {}'.format(len(stick_start_cont) / len(AA.indices_d['default'])))

        for axon_k in axon_far_times_d.keys():
            #print('BEFORE', len(events_after_stick[axon_k]))
            events_after_stick[axon_k] = np.array(list(set(events_after_stick[axon_k])))
            #print('AFTER', len(events_after_stick[axon_k]))
            print('Ratio {} stick events/all: {}'.format(axon_k, len(events_after_stick[axon_k])/len(AA.events_contained['default'][axon_k])))

    def get_after_stick_axon_events_plot(self, AA, max_delay=10, mode='axon_far', signal_side_len=100, max_stick_samples=40, max_signal_samples=20):
        '''
        Plot small plot of stick with axon events followed after (within the next second)
        '''
        axon_far_merged_events = []
        axon_far_merged_times = []

        if mode == 'axon_far':
            #Take all axon far from AA_l
            for axon_str in AA.axon_strs:
                k = axon_str + '_far'
                axon_far_merged_events.extend(AA.events_contained['default'][k])
                axon_far_merged_times.extend(AA.times_contained['default'][k])
        elif mode == 'axon_border':
            axon_far_merged_events.extend(AA.axon_bound_events_contained['default'])
            axon_far_merged_times.extend(AA.axon_bound_times_contained['default'])
        else:
            print('Mode {} not supported'.format(mode))
            return None

        axon_far_start_times = [axon_far_time[0] for axon_far_time in axon_far_merged_times]
        stick_start_bin = np.zeros([len(AA.stick_bin)])
        stick_start_bin[AA.indices_d['stick_exact_start']] = 1

        fig_l = []

        for stick_i in range(min(max_stick_samples, len(AA.indices_d['stick_exact_start']))):
            stick_index = AA.indices_d['stick_exact_start'][stick_i]

            fig_l.append([])

            y_stick = stick_start_bin[stick_index - signal_side_len: stick_index + signal_side_len]
            x_stick = np.arange(len(y_stick))

            print('Y STICK LEN', len(x_stick))
            if len(x_stick) > 0:
                fig_l[stick_i].append(plotly_utils.plot_scatter_fmt(x=x_stick, y=y_stick, astype='int'))

                delays_temp = axon_far_start_times - stick_index
                delays_temp_inds = np.where((delays_temp <= max_delay) & (delays_temp >= 0))

                for signal_i, delay_ind in enumerate(delays_temp_inds[0]):
                    delay = delays_temp[delay_ind]
                    ev = axon_far_merged_events[delay_ind]

                    print('delay : {} , event: {}'.format(delay, ev))
                    print('Stick i : {} {}, Event begin: {}'.format(stick_i, stick_index, axon_far_start_times[delay_ind]))
                    #TODO Can make a bit better considering the length of the signal itself but doesn't matter much
                    t_begin_local, t_end_local = self.event_i_time_segment(AA, ev, add_start=delay+signal_side_len, add_end=signal_side_len)

                    if t_begin_local is None:
                        continue

                    y_signal = AA.res_d['dff_only'][ev, t_begin_local:t_begin_local + (2*signal_side_len)]
                    x_signal = np.arange(len(y_signal))
                    fig_l[stick_i].append(plotly_utils.plot_scatter_fmt(x=x_signal, y=y_signal, mode='lines', title='scatter', astype='float', x_title='', y_title=''))

                    if max_signal_samples <= signal_i:
                        break
        return fig_l

    def get_axon_only_proportion_delays_plot(self, AA_l, min_delay=-20, max_delay=50, avg=False, gauss_f=False):
        '''
        Proportion delays for axon only events. All events within the astrocyte are excluded

        For stick find take stick_exact_start when the mouse first hits. Filter out any stick frames where mouse is
        not running before/after stick

        For running and rest:
            Take all rest frames. Stich them and then split into (max_delay-min_delay) frame segments
            Then see the events taking place at each point during the segment from min delay to max delay
        '''
        #Unique, no unique
        #Num stick start non num stick start
        unique_args = [True, False]
        max_duration_args = [None]

        figs = {}
        stick_id = 'stick_exact_start'
        running_id = 'running_exact'
        rest_id = 'rest_exact'

        #Split into max_delay-min_delay frames
        split_size = (max_delay - min_delay) + 1
        print('SPLIT SIZE?', split_size)
        running_prop, rest_prop = self.get_rest_run_proportion_events_interval(AA_l, running_id='running_exact', rest_id='rest_exact', interval=split_size, axon_bound_only=True)

        axon_border_events = {}
        for AA in AA_l:
            axon_border_events[AA.print_id] = AA.axon_bound_events_contained['default']

        for max_duration in max_duration_args:
            #STICK
            for un in unique_args:
                plot_id = 'prop-{}-{}'.format('unique' if un else 'notunique', 'max_duration_None' if (max_duration is None) else 'max_duration_' + str(max_duration))

                stick_prop = np.zeros([max_delay-min_delay+1])
                signal_delays_all_l = []

                for AA in AA_l:
                    #Filter hit stick indices to only when mouse is running during the period specified by min delay-max_delay
                    stick_indices_filt = aqua_utils.filter_range_inds(AA.indices_d[stick_id], AA.indices_d['running'], range=(min_delay, max_delay), prop=0.95)
                    print('LEN INDICES PROP: before {} after {}'.format(len(AA.indices_d[stick_id]), len(stick_indices_filt)))
                    delay_info_args = {'min_delay' : min_delay,
                                       'max_delay' : max_delay,
                                       'max_duration' : max_duration,
                                       'unique_events' : un,
                                       'event_inds_subset' : axon_border_events[AA.print_id]
                                    }
                    signal_delays_stick_np, peak_delays_stick_np = aqua_utils.get_delay_info_from_res(stick_indices_filt, AA.res_d, **delay_info_args)
                    signal_delays_all_l.extend(list(signal_delays_stick_np))

                signal_delays_all = np.array(signal_delays_all_l)
                for i, delay_x in enumerate(range(min_delay, max_delay+1)):
                    stick_prop[i] = float(np.sum(signal_delays_all == delay_x)) / len(signal_delays_all)

                x_l = [np.arange(min_delay, max_delay+1) for i in range(3)]
                y_l = [stick_prop, running_prop, rest_prop]

                def avg_x(x, avg=3):
                    new_x = []
                    for i in range((len(x) //  avg) -1):
                        print('x', i, x[avg*i])
                        new_x.append(x[avg*i])
                    return np.array(new_x)

                def avg_y(y, avg=3):
                    new_y = []
                    for i in range((len(y) // avg) - 1):
                        print('y', i, np.mean(y[i*avg:(i+1)*avg]))
                        new_y.append(np.mean(y[i*avg:(i+1)*avg]))

                    return np.array(new_y)
                if avg:
                    x_l = [avg_x(x_l[i]) for i in range(3)]
                    y_l = [avg_y(y_l[i]) for i in range(3)]
                if gauss_f:
                    x_l = x_l
                    y_l = [gaussian_filter(y_l[i], sigma=3) for i in range(3)]

                figs[plot_id] = plotly_utils.plot_scatter_mult(x_l, y_l, name_l=['stick', 'running', 'rest'], mode='lines', title='scatter', x_title='Delay (s)', y_title='Events')
                self.apply_fun_axis_fig(figs[plot_id], lambda x : x / AA.fr, axis='x')
        return figs

    def get_rest_run_proportion_events_interval(self, AA_l, running_id='running_exact', rest_id='rest_exact', interval=71, axon_bound_only=True):
        running_prop = np.zeros([interval])
        rest_prop = np.zeros([interval])

        for AA in AA_l:
            ############################################################
            #RUNNING AND REST
            running_ind = AA.indices_d[running_id]
            rest_ind = AA.indices_d[rest_id]

            if len(running_ind) % interval != 0:
                running_ind = running_ind[:-(len(running_ind) % interval)]

            if len(rest_ind) % interval != 0:
                rest_ind = rest_ind[:-(len(rest_ind) % interval)]

            running_split_l = np.split(running_ind, len(running_ind) / interval)
            rest_split_l = np.split(rest_ind, len(rest_ind) / interval)

            #Add events in delays based on their delay. Ignore events if there is max duration filter

            #For each split of frames, get events in those frames
            split_d = {'default' : AA.indices_d['default']}
            for i, running_split in enumerate(running_split_l):
                split_d['running_{}'.format(i)] = running_split
            for i, rest_split in enumerate(rest_split_l):
                split_d['rest_{}'.format(i)] = rest_split

            event_subsets, indices_events_bin = aqua_utils.get_event_subsets(split_d, AA.res_d, after_i=0, before_i=0, to_print=False, return_info=True)

            #print('SHAPE OF INDICES EVENTS:', indices_events_bin.shape)
            #print('AXON BOUND?', axon_bound_only)
            for k in split_d.keys():
                if k != 'default':
                    if axon_bound_only:
                        #Take subset of indices and also events that are axon bound
                        indices_events_k_subset = indices_events_bin[split_d[k], :]
                        indices_events_k_subset = indices_events_k_subset[:, AA.axon_bound_events_contained['default']]
                    else:
                        #Take subset of indices
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

        running_prop = running_prop / np.sum(running_prop)
        rest_prop = rest_prop / np.sum(rest_prop)
        return running_prop, rest_prop


    def get_transition_proportion_delays_plot_all(self, AA_l, before_bh, inds_bh, after_bh, before_delay=20, after_delay=50, before_range=20, after_range=50, avg_proportions=False):
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
        max_duration_args = [None]

        figs = {}

        for max_duration in max_duration_args:
            #STICK
            for un in unique_args:
                plot_id = 'prop-{}-{}'.format('unique' if un else 'notunique', 'max_duration_None' if (max_duration is None) else 'max_duration_' + str(max_duration))
                prop = np.zeros([after_range+before_range+1])
                signal_delays_all_l = []

                for AA in AA_l:
                    inds = AA.indices_d[inds_bh]
                    #Filter indices
                    indices_filt_before = aqua_utils.filter_range_inds(inds, AA.indices_d[before_bh], range=(-before_delay, -1), prop=1.0)
                    indices_filt_after = aqua_utils.filter_range_inds(inds, AA.indices_d[after_bh], range=(1, after_delay), prop=1.0)
                    indices_filt = np.array(np.sort(list(set(indices_filt_before) & set(indices_filt_after))))

                    if len(indices_filt) == 0:
                        continue
                    print('Len indices {} len filt before {} len filt after {} len filt {}'.format(len(inds), len(indices_filt_before), len(indices_filt_after), len(indices_filt)))

                    delay_info_args = {'min_delay' : -before_range,
                                   'max_delay' : after_range,
                                   'max_duration' : max_duration,
                                   'unique_events' : un,
                                   }

                    signal_delays_np, peak_delays_np = aqua_utils.get_delay_info_from_res(indices_filt, AA.res_d, **delay_info_args)
                    signal_delays_all_l.extend(list(signal_delays_np))

                signal_delays_all = np.array(signal_delays_all_l)
                for i, delay_x in enumerate(range(-before_range, after_range+1)):
                    prop[i] = float(np.sum(signal_delays_all == delay_x)) / len(signal_delays_all)

                x_l = [np.arange(-before_range, after_range+1) for i in range(1)]
                y_l = [prop]

                figs[plot_id] = plotly_utils.plot_scatter_mult(x_l, y_l, name_l=['{} to {}'.format(before_bh, after_bh)], mode='lines', title='scatter', x_title='Delay (s)', y_title='Events')
                self.apply_fun_axis_fig(figs[plot_id], lambda x : x / AA.fr, axis='x')
        return figs

    def get_axon_transition_proportion_delays_plot_all(self, AA_l, before_bh, inds_bh, after_bh,
                                                before_range=20, after_range=50, y_title=None,
                                                delay_step_size=1, fit=False, measure=None, fix_dff_interval=50, confidence=False,
                                                duration_filter=[None, None], setting='axon'):
        '''
        If setting is axon, get axon bound events (outside astrocyte)
        If setting is astro, get astrocyte landmark events
        '''
        figs = {}
        signal_delays_all_l_l = []

        axon_border_events = {}

        border_events = {}

        for AA in AA_l:
            if setting == 'axon':
                border_events[AA.print_id] = AA.axon_bound_events_contained['default']
            else:
                border_events[AA.print_id] = AA.astro_landmark_bound_events_contained['default']
        for AA in AA_l:
            inds = AA.indices_d[inds_bh]
            #Filter indices
            indices_filt_before = aqua_utils.filter_range_inds(inds, AA.indices_d[before_bh], range=(-before_range, -1), prop=1.0)
            indices_filt_after = aqua_utils.filter_range_inds(inds, AA.indices_d[after_bh], range=(1, after_range), prop=1.0)
            indices_filt = np.array(np.sort(list(set(indices_filt_before) & set(indices_filt_after))))

            #print('LEN INDICES_FILT: {}'.format(len(indices_filt)))
            #print('TOTAL IND {} BEFORE {} AFTER {} JOIN {}'.format(len(inds), len(indices_filt_before), len(indices_filt_after), len(indices_filt)))
            if len(indices_filt) == 0:
                continue
            #print('Len indices {} len filt before {} len filt after {} len filt {}'.format(len(inds), len(indices_filt_before), len(indices_filt_after), len(indices_filt)))
            #print('LEN INDICES FILT : {}'.format(len(indices_filt)))

            delay_info_args = {
                           'min_delay' : -before_range,
                           'max_delay' : after_range,
                           'min_duration' : duration_filter[0],
                           'max_duration' : duration_filter[1],
                           'unique_events' : False,
                           'return_non_unique_delays_arr' : True,
                           'event_inds_subset' : border_events[AA.print_id]
                           }

            _, _, _, signal_delays_l_l, peak_mins_l_l, valid_event_i_l_l = aqua_utils.get_delay_info_from_res(indices_filt, AA.res_d, **delay_info_args)

            for i, signal_delays_l in enumerate(signal_delays_l_l):
                signal_delays_all_l_l.append(signal_delays_l)

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
                    prop[i] = float(np.sum(np.array(signal_delays_all_l) == delay_x))
            prop_all_np[s_i, :] = prop

        #Working on proportion plots and event numbers
        prop_avg_events = np.sum(prop_all_np, axis=0) / (prop_all_np.shape[0])
        #print('BEFORE EVENTS', np.sum(np.sum(prop_all_np, axis=0)[0:before_range]))
        #print('AFTER EVENTS', np.sum(np.sum(prop_all_np, axis=0)[before_range:]))
        prop_avg_prop = np.sum(prop_all_np, axis=0) / np.sum(prop_all_np)
        prop_total_events = np.sum(prop_all_np, axis=0)

        bin_type = 'add' if measure is None else 'mean'
        #TODO HACK
        if delay_step_size != 1 and ((after_range + before_range) // delay_step_size != 2):
            bin_type = 'mean'

        x = np.arange(-before_range, after_range + 1, 1)

        print('CALLING FUNCTION HERE!!')
        fig, bin_stats = plotly_utils.plot_scatter_mult_tree(x=x, y_main=prop_avg_events, y_mult=prop_all_np, mode_main='lines', mode_mult='markers',
                                                    title='Average - Total events: {} Total intervals: {}'.format(total_events, prop_all_np.shape[0]),
                                                    y_title='Num events / interval' if y_title is None else y_title, x_title='Delay (s)', fit=fit, fit_annotation_pos_fix=AA.fr,
                                                    bin_main_size=delay_step_size, bin_mult_size=delay_step_size, opacity=0.5, confidence=confidence, with_stats=True,
                                                    bin_type=bin_type)

        print('BINS TATS HERE??', bin_stats.keys())
        confidence_format = 'lines' if delay_step_size == 1 else 'bar'
        fig2 = plotly_utils.plot_scatter_mult_tree(x=x, y_main=prop_avg_events, y_mult=prop_all_np, mode_main='lines', mode_mult='markers',
                                                    title='Average - Total events: {} Total intervals: {}'.format(total_events, prop_all_np.shape[0]),
                                                    y_title='Num events / interval' if y_title is None else y_title, x_title='Delay (s)', fit=fit, fit_annotation_pos_fix=AA.fr,
                                                    bin_main_size=delay_step_size, bin_mult_size=delay_step_size, opacity=0.5, confidence=confidence, y_mult_include=False,
                                                    confidence_format=confidence_format, bin_type=bin_type)


        #Normally we take the mean of the bin. However when we take the number of events in the bin
        #we want to add them up
        if len(x) // delay_step_size > 2:
            plotly_utils.apply_fun_axis_fig(fig, lambda x : x / AA.fr, axis='x')
            plotly_utils.apply_fun_axis_fig(fig2, lambda x : x / AA.fr, axis='x')

        #No proportions or total is used if we are doing measure
        fig3 = plotly_utils.plot_scatter(x=x, y=prop_avg_prop, title='Proportions - plot: Total events: {} Total intervals: {}'.format(total_events, prop_all_np.shape[0]),
                                                y_title='Normalized events (%)' if y_title is None else y_title, x_title='Delay (s)', bin_size=delay_step_size, bin_type=bin_type)

        plotly_utils.apply_fun_axis_fig(fig3, lambda x : x * 100, axis='y')

        if len(x) // delay_step_size > 2:
            plotly_utils.apply_fun_axis_fig(fig3, lambda x : x / AA.fr, axis='x')


        return {'event_avg_no_mult' : fig2,
                    'event_prop' : fig3}, bin_stats


    def get_axon_transition_proportion_delays_STICK_FILTER_plot_all(self, AA_l, before_bh, inds_bh, after_bh,
                                                before_range=20, after_range=50, y_title=None,
                                                delay_step_size=1, fit=False, measure=None, fix_dff_interval=50, confidence=False,
                                                duration_filter=[None, None], setting='axon'):
        '''
        If setting is axon, get axon bound events (outside astrocyte)
        If setting is astro, get astrocyte landmark events
        '''
        figs = {}
        signal_delays_all_l_l = []

        axon_border_events = {}

        border_events = {}

        for AA in AA_l:
            if setting == 'axon':
                border_events[AA.print_id] = AA.axon_bound_events_contained['default']
            else:
                border_events[AA.print_id] = AA.astro_landmark_bound_events_contained['default']
        for AA in AA_l:
            inds = AA.indices_d[inds_bh]
            #Filter indices
            indices_filt_before = aqua_utils.filter_range_inds(inds, AA.indices_d[before_bh], range=(-before_range, -1), prop=1.0)
            indices_filt_after = aqua_utils.filter_range_inds(inds, AA.indices_d[after_bh], range=(1, after_range), prop=1.0)
            indices_filt = np.array(np.sort(list(set(indices_filt_before) & set(indices_filt_after))))

            #print('LEN INDICES_FILT: {}'.format(len(indices_filt)))
            #print('TOTAL IND {} BEFORE {} AFTER {} JOIN {}'.format(len(inds), len(indices_filt_before), len(indices_filt_after), len(indices_filt)))
            if len(indices_filt) == 0:
                continue
            #print('Len indices {} len filt before {} len filt after {} len filt {}'.format(len(inds), len(indices_filt_before), len(indices_filt_after), len(indices_filt)))
            #print('LEN INDICES FILT : {}'.format(len(indices_filt)))

            delay_info_args = {
                           'min_delay' : -before_range,
                           'max_delay' : after_range,
                           'min_duration' : duration_filter[0],
                           'max_duration' : duration_filter[1],
                           'unique_events' : False,
                           'return_non_unique_delays_arr' : True,
                           'event_inds_subset' : border_events[AA.print_id]
                           }

            _, _, _, signal_delays_l_l, peak_mins_l_l, valid_event_i_l_l = aqua_utils.get_delay_info_from_res(indices_filt, AA.res_d, **delay_info_args)

            for i, signal_delays_l in enumerate(signal_delays_l_l):
                signal_delays_filt_l = []
                valid_event_i_l = valid_event_i_l_l[i]
                for j, valid_event_j in enumerate(valid_event_i_l):
                    #print('VALID EVENT I', valid_event_i)
                    #print('event id??', border_events[AA.print_id][valid_event_i])
                    event_id = border_events[AA.print_id][valid_event_j]
                    #print(indices_filt[i], AA.res_d['tBegin'][event_id], AA.res_d['tEnd'][event_id])
                    if event_id in AA.event_subsets['stick_exact_start']:
                        signal_delays_filt_l.append(signal_delays_l[j])
                print(len(signal_delays_filt_l))
                print('VS')
                print(len(signal_delays_l))
                signal_delays_all_l_l.append(signal_delays_filt_l)
                #signal_delays_all_l_l.append(signal_delays_l)

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
                    prop[i] = float(np.sum(np.array(signal_delays_all_l) == delay_x))
            prop_all_np[s_i, :] = prop

        #Working on proportion plots and event numbers
        prop_avg_events = np.sum(prop_all_np, axis=0) / (prop_all_np.shape[0])
        #print('BEFORE EVENTS', np.sum(np.sum(prop_all_np, axis=0)[0:before_range]))
        #print('AFTER EVENTS', np.sum(np.sum(prop_all_np, axis=0)[before_range:]))
        prop_avg_prop = np.sum(prop_all_np, axis=0) / np.sum(prop_all_np)
        prop_total_events = np.sum(prop_all_np, axis=0)

        bin_type = 'add' if measure is None else 'mean'
        #TODO HACK
        if delay_step_size != 1 and ((after_range + before_range) // delay_step_size != 2):
            bin_type = 'mean'

        x = np.arange(-before_range, after_range + 1, 1)

        print('CALLING FUNCTION HERE!!')
        fig, bin_stats = plotly_utils.plot_scatter_mult_tree(x=x, y_main=prop_avg_events, y_mult=prop_all_np, mode_main='lines', mode_mult='markers',
                                                    title='Average - Total events: {} Total intervals: {}'.format(total_events, prop_all_np.shape[0]),
                                                    y_title='Num events / interval' if y_title is None else y_title, x_title='Delay (s)', fit=fit, fit_annotation_pos_fix=AA.fr,
                                                    bin_main_size=delay_step_size, bin_mult_size=delay_step_size, opacity=0.5, confidence=confidence, with_stats=True,
                                                    bin_type=bin_type)

        print('BINS TATS HERE??', bin_stats.keys())
        confidence_format = 'lines' if delay_step_size == 1 else 'bar'
        fig2 = plotly_utils.plot_scatter_mult_tree(x=x, y_main=prop_avg_events, y_mult=prop_all_np, mode_main='lines', mode_mult='markers',
                                                    title='Average - Total events: {} Total intervals: {}'.format(total_events, prop_all_np.shape[0]),
                                                    y_title='Num events / interval' if y_title is None else y_title, x_title='Delay (s)', fit=fit, fit_annotation_pos_fix=AA.fr,
                                                    bin_main_size=delay_step_size, bin_mult_size=delay_step_size, opacity=0.5, confidence=confidence, y_mult_include=False,
                                                    confidence_format=confidence_format, bin_type=bin_type)


        #Normally we take the mean of the bin. However when we take the number of events in the bin
        #we want to add them up
        if len(x) // delay_step_size > 2:
            plotly_utils.apply_fun_axis_fig(fig, lambda x : x / AA.fr, axis='x')
            plotly_utils.apply_fun_axis_fig(fig2, lambda x : x / AA.fr, axis='x')

        #No proportions or total is used if we are doing measure
        fig3 = plotly_utils.plot_scatter(x=x, y=prop_avg_prop, title='Proportions - plot: Total events: {} Total intervals: {}'.format(total_events, prop_all_np.shape[0]),
                                                y_title='Normalized events (%)' if y_title is None else y_title, x_title='Delay (s)', bin_size=delay_step_size, bin_type=bin_type)

        plotly_utils.apply_fun_axis_fig(fig3, lambda x : x * 100, axis='y')

        if len(x) // delay_step_size > 2:
            plotly_utils.apply_fun_axis_fig(fig3, lambda x : x / AA.fr, axis='x')


        return {'event_avg_no_mult' : fig2,
                    'event_prop' : fig3}, bin_stats






    def create_landmark_distances_csv(self, AA, csv_path):
        distances_d = {}
        for k0 in AA.landmark_centroids.keys():
            for k1 in AA.landmark_centroids.keys():
                dist_raw = aqua_utils.get_euclidean_distances(AA.landmark_centroids[k0], AA.landmark_centroids[k1])[0]
                dist_um = dist_raw * (AA.spatial_res**2)
                distances_d['{}-{}'.format(k0, k1)] = general_utils.truncate(dist_um, 2)

        with open(csv_path, mode='w') as csv_file:
            writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['landmark_i-landmark_j', 'distance (um)'])

            for k in distances_d.keys():
                writer.writerow([k, distances_d[k]])





















    ################################################################################
    ################################################################################
    ################################################################################
    ################################################################################
    #NOT USED ANYMORE

    """
    def get_axon_only_proportion_delays_plot_old(self, AA_l, min_delay=-20, max_delay=50, avg=False, gauss_f=False):
        '''
        Plot waterfall of latencies for only axon events
        '''
        #Go through every event and check if it is contained in the axon only area bound
        unique_args = [True, False]
        max_duration_args = [None]
        with_stick_num_args = [True]

        figs = {}
        figs_interp = {}

        stick_id = 'stick_exact_start'
        running_id = 'running_exact'
        rest_id = 'rest_exact'

        #Get list of events for each axon far in each AstroAxon
        axon_border_events = {}

        for AA in AA_l:
            axon_border_events[AA.print_id] = AA.axon_bound_events_contained['default']

        for AA in AA_l:
            for un in unique_args:
                for max_duration in max_duration_args:
                    for with_stick_num in with_stick_num_args:
                        delay_info_args = {'min_delay' : min_delay,
                                           'max_delay' : max_delay,
                                           'max_duration' : max_duration,
                                           'unique_events' : un,
                                           'event_inds_subset' : axon_border_events[AA.print_id]
                                        }
                        plot_id = '{}-{}-{}-{}'.format('unique' if un else 'notunique',
                                                    'max_duration_None' if (max_duration is None) else 'max_duration_' + str(max_duration),
                                                    'stick_num_' + str(with_stick_num),
                                                    AA.print_id)
                        if with_stick_num:
                            rand_running = np.random.choice(list(set(AA.indices_d[running_id]) - set(AA.indices_d[stick_id])), size=len(AA.indices_d[stick_id]), replace=False)
                            rand_no_running = np.random.choice(list(set(AA.indices_d[rest_id]) - set(AA.indices_d[stick_id])), size=len(AA.indices_d[stick_id]), replace=False)
                        else:
                            rand_running = list(set(AA.indices_d[running_id]) - set(AA.indices_d[stick_id]))
                            rand_no_running = list(set(AA.indices_d[rest_id]) - set(AA.indices_d[stick_id]))

                        signal_delays_stick_np, peak_delays_stick_np = aqua_utils.get_delay_info_from_res(AA.indices_d[stick_id], AA.res_d, **delay_info_args)
                        signal_delays_running_np, peak_delays_running_np = aqua_utils.get_delay_info_from_res(rand_running, AA.res_d, **delay_info_args)
                        signal_delays_no_running_np, peak_delays_no_running_np = aqua_utils.get_delay_info_from_res(rand_no_running, AA.res_d, **delay_info_args)

                    stick_prop = np.zeros([max_delay-min_delay+1])
                    running_prop = np.zeros([max_delay-min_delay+1])
                    no_running_prop = np.zeros([max_delay-min_delay+1])

                    for i, delay_x in enumerate(range(min_delay, max_delay+1)):
                        stick_prop[i] = float(np.sum(signal_delays_stick_np == delay_x)) / len(signal_delays_stick_np)
                        running_prop[i] = float(np.sum(signal_delays_running_np == delay_x)) / len(signal_delays_running_np)
                        no_running_prop[i] = float(np.sum(signal_delays_no_running_np == delay_x)) / len(signal_delays_no_running_np)

                    x_l = [np.arange(min_delay, max_delay+1) for i in range(3)]
                    y_l = [stick_prop, running_prop, no_running_prop]

                    def avg_x(x, avg=3):
                        new_x = []
                        for i in range((len(x) //  avg) -1):
                            print('x', i, x[avg*i])
                            new_x.append(x[avg*i])
                        return np.array(new_x)

                    def avg_y(y, avg=3):
                        new_y = []
                        for i in range((len(y) // avg) - 1):
                            print('y', i, np.mean(y[i*avg:(i+1)*avg]))
                            new_y.append(np.mean(y[i*avg:(i+1)*avg]))

                        return np.array(new_y)
                    if avg:
                        x_l = [avg_x(x_l[i]) for i in range(3)]
                        y_l = [avg_y(y_l[i]) for i in range(3)]
                    if gauss_f:
                        x_l = x_l
                        y_l = [gaussian_filter(y_l[i], sigma=3) for i in range(3)]

                    figs[plot_id] = plotly_utils.plot_scatter_mult(x_l, y_l, name_l=['stick', 'running', 'rest'], mode='lines', title='scatter', x_title='Delay (s)', y_title='Events')
                    self.apply_fun_axis_fig(figs[plot_id], lambda x : x / AA.fr, axis='x')
        return figs



    def get_waterfall_delays_axons_far_plot(self, AA_l, return_results_only=False):
        unique_args = [True, False]
        max_duration_args = [None]
        with_stick_num_args = [True]

        figs = {}
        figs_interp = {}

        stick_id = 'stick_exact_start'
        running_id = 'running_exact'
        rest_id = 'rest_exact'

        stick_v_l_d = {}
        running_v_l_d = {}
        no_running_v_l_d = {}

        #Get list of events for each axon far in each AstroAxon
        axon_far_events_d = {}
        axon_far_merged_events_d = {}
        for AA in AA_l:
            axon_far_events_d[AA.print_id] = {}
            axon_far_merged_events_d[AA.print_id] = []
            #Take all axon far from AA_l
            for axon_str in AA.axon_strs:
                k = axon_str + '_far'
                axon_far_events_d[AA.print_id][k] = AA.events_contained['default'][k]
                axon_far_merged_events_d[AA.print_id].extend(AA.events_contained['default'][k])

        print('AXON FAR EVENTS', axon_far_events_d)
        print('axon far merged events', axon_far_merged_events_d)

        for AA_i, AA in enumerate(AA_l):
            for un in unique_args:
                for max_duration in max_duration_args:
                    for with_stick_num in with_stick_num_args:
                        delay_info_args = {'min_delay' : -20,
                                           'max_delay' : 50,
                                           'max_duration' : max_duration,
                                           'unique_events' : un,
                                           'event_inds_subset' : axon_far_merged_events_d[AA.print_id]
                                        }
                        plot_id = '{}-{}-{}-{}'.format('unique' if un else 'notunique',
                                                    'max_duration_None' if (max_duration is None) else 'max_duration_' + str(max_duration),
                                                    'stick_num_' + str(with_stick_num),
                                                    AA.print_id)
                        plot_id_2 = '{}-{}-{}'.format('unique' if un else 'notunique',
                                                    'max_duration_None' if (max_duration is None) else 'max_duration_' + str(max_duration),
                                                    'stick_num_' + str(with_stick_num))
                        if AA_i == 0:
                            stick_v_l_d[plot_id_2] = []
                            running_v_l_d[plot_id_2] = []
                            no_running_v_l_d[plot_id_2] = []

                        if with_stick_num:
                            rand_running = np.random.choice(list(set(AA.indices_d[running_id]) - set(AA.indices_d[stick_id])), size=len(AA.indices_d[stick_id]), replace=False)
                            rand_no_running = np.random.choice(list(set(AA.indices_d[rest_id]) - set(AA.indices_d[stick_id])), size=len(AA.indices_d[stick_id]), replace=False)
                        else:
                            rand_running = list(set(AA.indices_d[running_id]) - set(AA.indices_d[stick_id]))
                            rand_no_running = list(set(AA.indices_d[rest_id]) - set(AA.indices_d[stick_id]))

                        signal_delays_stick_np, peak_delays_stick_np = aqua_utils.get_delay_info_from_res(AA.indices_d[stick_id], AA.res_d, **delay_info_args)
                        signal_delays_running_np, peak_delays_running_np = aqua_utils.get_delay_info_from_res(rand_running, AA.res_d, **delay_info_args)
                        signal_delays_no_running_np, peak_delays_no_running_np = aqua_utils.get_delay_info_from_res(rand_no_running, AA.res_d, **delay_info_args)

                        stick_v = np.sort(signal_delays_stick_np)
                        running_v = np.sort(signal_delays_running_np)
                        no_running_v = np.sort(signal_delays_no_running_np)

                        stick_v_l_d[plot_id_2].extend(stick_v)
                        running_v_l_d[plot_id_2].extend(running_v)
                        no_running_v_l_d[plot_id_2].extend(no_running_v)

                        figs[plot_id] = plotly_utils.plot_waterfall(arrays_l=[stick_v, running_v, no_running_v], legend_names=['stick', 'running', 'rest'], title='Signal (event) delays after behaviour', x_title='Delay (s)', y_title='Event id')
                        self.apply_fun_axis_fig(figs[plot_id], lambda x : x / AA.fr, axis='x')
                        figs_interp[plot_id] = plotly_utils.plot_waterfall_interpolate(arrays_l=[stick_v, running_v, no_running_v], legend_names=['stick', 'running', 'rest'], title='Signal (event) delays after behaviour (scaled)', x_title='Delay (s)', y_title='Event id')
                        self.apply_fun_axis_fig(figs_interp[plot_id], lambda x : x / AA.fr, axis='x', )

        for k in stick_v_l_d.keys():
            stick_v_l_d[k] = np.sort(stick_v_l_d[k])
            running_v_l_d[k] = np.sort(running_v_l_d[k])
            no_running_v_l_d[k] = np.sort(no_running_v_l_d[k])

        if return_results_only:
            return [stick_v_l_d, running_v_l_d, no_running_v_l_d]
        return figs, figs_interp

    def get_waterfall_delays_together_axons_far_plot(self, AA_l):
        stick_v_l_d, running_v_l_d, no_running_v_l_d = self.get_waterfall_delays_axons_far_plot(AA_l, return_results_only=True)

        figs_d = {}
        figs_interp_d = {}

        for k in stick_v_l_d.keys():
            stick_v = stick_v_l_d[k]
            running_v = running_v_l_d[k]
            no_running_v = no_running_v_l_d[k]

            fig = plotly_utils.plot_waterfall(arrays_l=[stick_v, running_v, no_running_v], legend_names=['stick', 'running', 'rest'], title='Signal (event) delays after behaviour', x_title='Delay (s)', y_title='Event id')
            self.apply_fun_axis_fig(fig, lambda x : x / AA_l[0].fr, axis='x')
            fig_interp = plotly_utils.plot_waterfall_interpolate(arrays_l=[stick_v, running_v, no_running_v], legend_names=['stick', 'running', 'rest'], title='Signal (event) delays after behaviour (scaled) All axons', x_title='Delay (s)', y_title='Event id')
            self.apply_fun_axis_fig(fig_interp, lambda x : x / AA_l[0].fr, axis='x')

            figs_d[k] = fig
            figs_interp_d[k] = fig_interp
        return figs_d, figs_interp_d

    def get_axon_only_bound_waterfall_latencies_plot(self, AA_l):
        '''
        Plot waterfall of latencies for only axon events
        '''
        #Go through every event and check if it is contained in the axon only area bound
        unique_args = [True, False]
        max_duration_args = [None]
        with_stick_num_args = [True]

        figs = {}
        figs_interp = {}

        stick_id = 'stick_exact_start'
        running_id = 'running_exact'
        rest_id = 'rest_exact'

        #Get list of events for each axon far in each AstroAxon
        axon_border_events = {}

        for AA in AA_l:
            axon_border_events[AA.print_id] = AA.axon_bound_events_contained['default']

        for AA in AA_l:
            for un in unique_args:
                for max_duration in max_duration_args:
                    for with_stick_num in with_stick_num_args:
                        delay_info_args = {'min_delay' : -20,
                                           'max_delay' : 50,
                                           'max_duration' : max_duration,
                                           'unique_events' : un,
                                           'event_inds_subset' : axon_border_events[AA.print_id]
                                        }
                        plot_id = '{}-{}-{}-{}'.format('unique' if un else 'notunique',
                                                    'max_duration_None' if (max_duration is None) else 'max_duration_' + str(max_duration),
                                                    'stick_num_' + str(with_stick_num),
                                                    AA.print_id)
                        if with_stick_num:
                            rand_running = np.random.choice(list(set(AA.indices_d[running_id]) - set(AA.indices_d[stick_id])), size=len(AA.indices_d[stick_id]), replace=False)
                            rand_no_running = np.random.choice(list(set(AA.indices_d[rest_id]) - set(AA.indices_d[stick_id])), size=len(AA.indices_d[stick_id]), replace=False)
                        else:
                            rand_running = list(set(AA.indices_d[running_id]) - set(AA.indices_d[stick_id]))
                            rand_no_running = list(set(AA.indices_d[rest_id]) - set(AA.indices_d[stick_id]))

                        signal_delays_stick_np, peak_delays_stick_np = aqua_utils.get_delay_info_from_res(AA.indices_d[stick_id], AA.res_d, **delay_info_args)
                        signal_delays_running_np, peak_delays_running_np = aqua_utils.get_delay_info_from_res(rand_running, AA.res_d, **delay_info_args)
                        signal_delays_no_running_np, peak_delays_no_running_np = aqua_utils.get_delay_info_from_res(rand_no_running, AA.res_d, **delay_info_args)

                        stick_v = np.sort(signal_delays_stick_np)
                        running_v = np.sort(signal_delays_running_np)
                        no_running_v = np.sort(signal_delays_no_running_np)

                        figs[plot_id] = plotly_utils.plot_waterfall(arrays_l=[stick_v, running_v, no_running_v], legend_names=['stick', 'running', 'rest'], title='Signal (event) delays after behaviour', x_title='Delay (s)', y_title='Event id')
                        self.apply_fun_axis_fig(figs[plot_id], lambda x : x / AA.fr, axis='x')
                        figs_interp[plot_id] = plotly_utils.plot_waterfall_interpolate(arrays_l=[stick_v, running_v, no_running_v], legend_names=['stick', 'running', 'rest'], title='Signal (event) delays after behaviour (scaled)', x_title='Delay (s)', y_title='Event id')
                        self.apply_fun_axis_fig(figs_interp[plot_id], lambda x : x / AA.fr, axis='x', )
                        print('added?')
        print('FIGS', figs)
        print('INRERP', figs_interp)
        return figs, figs_interp

    """


    #This is literally copy paste from AstrocytePlotter but whatevs
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
