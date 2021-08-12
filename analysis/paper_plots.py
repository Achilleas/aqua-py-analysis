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
from collections import deque
import powerlaw
import pylab
from matplotlib.font_manager import FontProperties
from matplotlib import rc
from scipy import stats
from scipy.stats import skewnorm
import plotly.graph_objs as go

def generate_astro_single_plots(astro_plotter, astroA, output_folder):
    output_experiment_path = astro_plotter.get_output_experiment_path(astroA, output_folder)

    print('Plotting behaviours basic...')
    #Behaviour basic
    figs_basic_plots = astro_plotter.get_behaviour_basic_plots(astroA)
    for fig_k in figs_basic_plots.keys():
        saving_utils.save_plotly_fig(figs_basic_plots[fig_k], os.path.join(output_experiment_path, 'plots', 'behaviours_basic', '{}'.format(fig_k)), width=1000, height=400)


    print('Plotting behaviour heatmaps...')
    #Behaviour heatmaps
    fig_heatmap_grids, fig_heatmap_dff_grids = astro_plotter.get_behaviour_contour_plots(astroA)
    heatmap_grid_base_path = os.path.join(output_experiment_path, 'plots', 'behaviour_heatmaps')
    for k in fig_heatmap_grids.keys():
        saving_utils.save_plotly_fig(fig_heatmap_grids[k], os.path.join(heatmap_grid_base_path, k))
        saving_utils.save_plotly_fig(fig_heatmap_dff_grids[k], os.path.join(heatmap_grid_base_path, k + 'dff'))

    print('Plotting behaviour heatmaps (saturation)...')
    fig_heatmap_grids, fig_heatmap_dff_grids = astro_plotter.get_behaviour_contour_threshold_plots(astroA, threshold=0.5)
    heatmap_grid_base_path = os.path.join(output_experiment_path, 'plots', 'behaviour_heatmaps_saturation')
    for k in fig_heatmap_grids.keys():
        saving_utils.save_plotly_fig(fig_heatmap_grids[k], os.path.join(heatmap_grid_base_path, k))
        saving_utils.save_plotly_fig(fig_heatmap_dff_grids[k], os.path.join(heatmap_grid_base_path, k + 'dff'))

    print('Plotting borders...')
    #Borders plot
    fig_border = astro_plotter.get_border_plot(astroA)
    saving_utils.save_plotly_fig(fig_border, os.path.join(output_experiment_path, 'plots' , 'borders', 'border'))

    print('Plotting behaviour activity bar plot...')
    behaviour_activity_path = os.path.join(output_experiment_path, 'plots', 'behaviour_activity', 'activity')
    fig_behaviour_activity = astro_plotter.get_behaviour_activity_plot(astroA)
    saving_utils.save_plotly_fig(fig_behaviour_activity, behaviour_activity_path, width=1200, height=800)

    print('Plotting behaviour event size bar plot...')
    behaviour_area_path = os.path.join(output_experiment_path, 'plots', 'behaviour_areas', 'areas')
    fig_behaviour_area = astro_plotter.get_behaviour_area_plot(astroA)
    saving_utils.save_plotly_fig(fig_behaviour_area, behaviour_area_path)

    print('Plotting behaviour amplitude size bar plot...')
    behaviour_amplitude_path = os.path.join(output_experiment_path, 'plots', 'signal_amplitudes', 'amplitudes')
    fig_behaviour_amplitude = astro_plotter.get_behaviour_amplitude_bar_plot(astroA)
    saving_utils.save_plotly_fig(fig_behaviour_amplitude, behaviour_amplitude_path)


def generate_astro_comparison_plots(astro_plotter, astroA_l, output_folder, name_tag, astroA_l_pairs=None, astroA_long_l=None, n_chunks=3):
    output_experiment_path_all_comparison, _, _, astroA_l_s = astro_plotter.setup_comparison_all_vars(astroA_l, os.path.join(output_folder, name_tag))

    print('Plotting sizes histogram dataset comparison for each behaviour')

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

    print('Comparing behaviour distribution plots...')

    configs = [
        {'measure': 'area', 'range': [None, 60], 'nbins' : 20, 'bh_l' : ['rest', 'stick_rest', 'running', 'stick_run_ind_15'], 'mode' : 'MOE'},
        {'measure': 'dffMax2', 'range': [0.6, 5], 'nbins' : 20, 'bh_l' : ['rest', 'stick_rest', 'running', 'stick_run_ind_15'], 'mode' : 'MOE'},
        {'measure': 'duration', 'range' : [None, 30], 'nbins' : 10, 'bh_l' : ['rest', 'stick_rest', 'running', 'stick_run_ind_15'], 'mode' : 'MOA'}
    ]

    for config in configs:
        behaviour_l = config['bh_l']
        measure = config['measure']
        min_measure, max_measure = config['range']
        mode = config['mode']
        n_bins = config['nbins']
        confidence = True

        try:
            measure_name = aqua_utils.get_measure_names(measure)
            path = os.path.join(output_experiment_path_all_comparison, 'plots', '{}_histogram_bh_comparison'.format(measure_name), 'behaviours-{}-nbins={}-min={}-max={}-conf={}-mode={}'.format('_'.join(behaviour_l), n_bins, min_measure, max_measure, confidence, mode))
            plot, stats_d = astro_plotter.measure_distribution_bh_compare_plot(astroA_l, behaviour_l, measure=measure, num_bins=n_bins, min_measure=min_measure, max_measure=max_measure, measure_name=measure_name, confidence=confidence, with_stats=True, mode=mode)

            if measure == 'duration':
                plotly_utils.apply_fun_axis_fig(plot, lambda x : x / astroA_l[0].fr, axis='x')

            if measure == 'area':
                saving_utils.save_pth_plt_l_log([plot], [path], axis='x')
            else:
                saving_utils.save_plotly_fig(plot, path)
            #saving_utils.save_pth_plt_l_log([plot], [path], axis='y')

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
        except Exception as e:
            print('Exception: {}'.format(e))

    #------------------------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------

    delay_ranges_pairs = [[3*astroA_l[0].fr, 6*astroA_l[0].fr], [2*astroA_l[0].fr, 4*astroA_l[0].fr]]
    delay_ranges_pairs = [[int(v[0]), int(v[1])] for v in delay_ranges_pairs]
    before_range_3, after_range_6 = delay_ranges_pairs[0]
    before_range_2, after_range_4 = delay_ranges_pairs[1]

    print('Alt Proportion plots...')

    # Rest to run plots
    rest_to_run_setting = { 
                'before_bh':'rest_semi_exact', 
                'inds_bh':'running_exact_start', 
                'after_bh':'running_semi_exact', 
                'before_range' : before_range_3,
                'after_range' : after_range_6,
                'fit': False, 
                'delay_step_size': 10,
                'confidence': True}

    # Rest to run - PROPORTIONS
    
    __save_astro_transition_plot(astro_plotter, astroA_l, setting=rest_to_run_setting, plot_type='proportions', 
        path=os.path.join(output_experiment_path_all_comparison, 'plots', 'rest_to_run_proportions'))
    
    __save_astro_transition_plot(astro_plotter, astroA_l, setting=rest_to_run_setting, plot_type='measure', measure='dffMax2default',
        path=os.path.join(output_experiment_path_all_comparison, 'plots', 'rest_to_run_amplitudes'))
    __save_astro_transition_plot(astro_plotter, astroA_l, setting=rest_to_run_setting, plot_type='measure', measure='time_s',
        path=os.path.join(output_experiment_path_all_comparison, 'plots', 'rest_to_run_durations'))
    __save_astro_transition_plot(astro_plotter, astroA_l, setting=rest_to_run_setting, plot_type='measure', measure='area',
        path=os.path.join(output_experiment_path_all_comparison, 'plots', 'rest_to_run_sizes'))

    rest_to_run_setting['delay_step_size'] = 5
    __save_astro_transition_plot(astro_plotter, astroA_l, setting=rest_to_run_setting, plot_type='behaviour', bh_measure='speed',
        path=os.path.join(output_experiment_path_all_comparison, 'plots', 'rest_to_run_speed'))

    # Run to rest plots
    run_to_rest_setting = { 
                'before_bh':'running_semi_exact', 
                'inds_bh':'rest_start', 
                'after_bh':'rest_semi_exact', 
                'before_range' : before_range_3,
                'after_range' : after_range_6,
                'fit': False, 
                'delay_step_size': 10,
                'confidence': True}

    __save_astro_transition_plot(astro_plotter, astroA_l, setting=run_to_rest_setting, plot_type='proportions',
        path=os.path.join(output_experiment_path_all_comparison, 'plots', 'run_to_rest_proportions'))
    __save_astro_transition_plot(astro_plotter, astroA_l, setting=run_to_rest_setting, plot_type='measure', measure='dffMax2default',
        path=os.path.join(output_experiment_path_all_comparison, 'plots', 'run_to_rest_amplitudes'))
    __save_astro_transition_plot(astro_plotter, astroA_l, setting=run_to_rest_setting, plot_type='measure', measure='time_s',
        path=os.path.join(output_experiment_path_all_comparison, 'plots', 'run_to_rest_durations'))
    __save_astro_transition_plot(astro_plotter, astroA_l, setting=run_to_rest_setting, plot_type='measure', measure='area',
        path=os.path.join(output_experiment_path_all_comparison, 'plots', 'run_to_rest_sizes'))

    run_to_rest_setting['delay_step_size'] = 5
    __save_astro_transition_plot(astro_plotter, astroA_l, setting=run_to_rest_setting, plot_type='behaviour', bh_measure='speed',
        path=os.path.join(output_experiment_path_all_comparison, 'plots', 'run_to_rest_speed'))

    # Run-stick-run plots
    run_stick_run_setting = {
                'before_bh':'running_semi_exact', 
                'inds_bh':'stick_exact_start', 
                'after_bh':'running_semi_exact', 
                'before_range' : before_range_2,
                'after_range' : after_range_4,
                'fit': False, 
                'delay_step_size': 10,
                'confidence': True}

    __save_astro_transition_plot(astro_plotter, astroA_l, setting=run_stick_run_setting, plot_type='proportions',
        path=os.path.join(output_experiment_path_all_comparison, 'plots', 'run_stick_run_proportions'))
    __save_astro_transition_plot(astro_plotter, astroA_l, setting=run_stick_run_setting, plot_type='measure', measure='dffMax2default',
        path=os.path.join(output_experiment_path_all_comparison, 'plots', 'run_stick_run_amplitudes'))
    __save_astro_transition_plot(astro_plotter, astroA_l, setting=run_stick_run_setting, plot_type='measure', measure='time_s',
        path=os.path.join(output_experiment_path_all_comparison, 'plots', 'run_stick_run_durations'))
    __save_astro_transition_plot(astro_plotter, astroA_l, setting=run_stick_run_setting, plot_type='measure', measure='area',
        path=os.path.join(output_experiment_path_all_comparison, 'plots', 'run_stick_run_sizes'))

    __save_astro_transition_plot(astro_plotter, astroA_l, setting=run_stick_run_setting, plot_type='behaviour', bh_measure='speed',
        path=os.path.join(output_experiment_path_all_comparison, 'plots', 'run_stick_run_speed'))
    
    #------------------------------------------------------------------------------------------------------------------

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
        hist = hist * (bin_edges[1] - bin_edges[0])
        print('HIST SUM', np.sum(hist))
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
        d = astro_plotter.get_individual_heatmaps_threshold_scaled(astroA, bh='default', threshold=1, num_samples=1, dff_mode=False, with_arr=True)
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
        hist = hist * (bin_edges[1] - bin_edges[0])
        print('HIST SUM', np.sum(hist))
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


    #------------------------------------------------------------------------------------------------------------------

    path = os.path.join(output_experiment_path_all_comparison, 'plots', 'power_law_fit_sizes_distribution')
    path = path +'/'
    saving_utils.generate_directory_path(path)
    pylab.rcParams['xtick.major.pad']='8'
    pylab.rcParams['ytick.major.pad']='8'

    rc('font', family='sans-serif')
    rc('font', size=10.0)
    rc('text', usetex=False)

    panel_label_font = FontProperties().copy()
    panel_label_font.set_weight("bold")
    panel_label_font.set_size(12.0)
    panel_label_font.set_family("sans-serif")

    fig, x, y_l, all_events_measure_l = astro_plotter.measure_distribution_plot(astroA_l, 'default', 'area', num_bins=10, min_measure=None, max_measure=None, measure_name='area', mode='MOE', with_measure_values=True)

    xmin=5
    data_np = np.array(all_events_measure_l)
    fit = powerlaw.Fit(data_np, discrete=True, xmin=xmin)
    ####
    fig = fit.plot_ccdf(linewidth=3, label='Empirical Data')
    fit.power_law.plot_ccdf(ax=fig, color='r', linestyle='--', label='Power law fit')
    fit.lognormal.plot_ccdf(ax=fig, color='g', linestyle='--', label='Lognormal fit')
    fit.exponential.plot_ccdf(ax=fig, color='b', linestyle='--', label='Exponential fit')
    ####

    fig.set_ylabel(u"p(X≥x)")
    fig.set_xlabel("Size µm^2")
    handles, labels = fig.get_legend_handles_labels()
    fig.legend(handles, labels, loc=3)

    figname = 'EmpiricalvsFits'

    plt.savefig(os.path.join(path, figname+'.svg'), bbox_inches='tight')
    plt.savefig(os.path.join(path, figname+'.png'), bbox_inches='tight')

    #print('POWER LAW VS LOG NORMAL', fit.distribution_compare('power_law', 'lognormal'))
    #print('POWER LAW VS EXPONENTIAL cutoff at {}µm**2'.format(xmin), fit.distribution_compare('power_law', 'exponential'))
    #print('POWERLAW FUNCTION: ~x**(-{})'.format(fit.power_law.alpha))



    #------------------------------------------------------------------------------------------------------------------

    plt.ioff()
    print('Plotting Size vs Time correlation plot...')
    path = os.path.join(output_experiment_path_all_comparison, 'plots', 'size_v_time_corr_ALL')
    path = path+'/'
    print('Generating direcotyr path', path + '/')
    saving_utils.generate_directory_path(path)
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


    #------------------------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------    
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
    #4 (correlation between split days with variable the rest-run behaviour)
    for bh in ['rest']:
        #2
        fig, res_splits_l = astro_plotter.get_between_split_split_xcorr(astroA_long_l, bh=bh, save_pkl_path=save_splits_pkl_path, n_chunks=n_chunks)
        #3
        fig_2, res_day_splits_l = astro_plotter.get_between_day_split_xcorr(day_0_1_pairs, bh=bh, save_pkl_path=save_day_splits_pkl_path, n_chunks=n_chunks)
        #4
        fig_3, res_bh_splits_l = astro_plotter.get_between_bh_split_xcorr(astroA_long_l, bh_pair=['rest','running'], save_pkl_path=save_bh_splits_pkl_path, n_chunks=n_chunks)
        #1
        if os.path.isfile(save_random_pkl_path):
            random_l = saving_utils.load_pickle(save_random_pkl_path)
        else:
            random_l = []
            for astroA in astroA_long_l:
                random_l.extend(astro_plotter.get_random_corrs_self(astroA, bh, n_fake_samples=3))
        if save_random_pkl_path is not None:
            saving_utils.save_pickle(random_l, save_random_pkl_path)

        x = ['Random', 'Self splits', 'Rest-Run splits', 'Day 0-1 Splits']
        y = [random_l, res_splits_l, res_bh_splits_l, res_day_splits_l]

        fig, stats_d = plotly_utils.plot_point_box_revised(x, y, title='Split correlations (between splits)- {}'.format(bh), x_title='', y_title='Xcorr value', with_stats=True)
        saving_utils.save_plotly_fig(fig, os.path.join(plot_folder, 'splits'))
        
        saving_utils.dict_to_csv(stats_d, os.path.join(plot_folder, 'splits' + '.csv'))

        #saving_utils.save_csv_dict(stats_d, os.path.join(plot_folder, 'splits' + '.csv'), key_order=['x', 'mean', 'conf_95'])

        results_dict = {x[i] : y[i] for i in range(len(x))}
        saving_utils.dict_to_csv(results_dict, os.path.join(plot_folder, 'splits-data' + '.csv'))

        
        #results_dict['x'] = x
        #key_order = ['x']
        #key_order.extend(x)
        #saving_utils.save_csv_dict(results_dict, os.path.join(plot_folder, 'splits_data' + '.csv'), key_order=key_order)

    #------------------------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------
    print('Plotting correlation of self splitted plots...')
    #STEP 1
    #Take only long duration astrocytes
    #Set maximum length of astrocyte duration to be 70min
    #Then apply splits with xcorr
    data_save_path = os.path.join(output_experiment_path_all_comparison, 'data', 'splits_self_all')
    path = os.path.join(output_experiment_path_all_comparison, 'plots', 'splits_self_all', 'self_all')
    y_l_l = []
    x_l = []
    minute_frame_splits_l = [35, 30, 25, 20, 15, 10, 5, 2]
    cut_duration = 70
    param_str = 'cut_{}-'.format(cut_duration) + 'splits_{}-'.format('_'.join([str(m) for m in minute_frame_splits_l]))

    name_l = []
    for i, astroA in enumerate(astroA_long_l):
        curr_save_path = os.path.join(data_save_path, 'id_{}-{}.pkl'.format(astroA.print_id, param_str))
        res_d = astro_plotter.get_compare_full_self_results_alt(astroA, cut_duration_min=cut_duration, minute_frame_splits_l=minute_frame_splits_l, save_pkl_path=curr_save_path)
        y_l_l.append(res_d['y'])
        x_l.append(res_d['x'])
        name_l.append(astroA.print_id)

    fig, stats_d = plotly_utils.plot_scatter_mult_with_avg(x_l[0], y_l_l, None, name_l, mode='lines', title='Splits self', x_title='Splits (minutes)', y_title='Correlation',
                    xrange=None, yrange=None, confidence=True, with_stats=True, point_box=True, exclude_non_avg_conf=True)

    print(path)
    saving_utils.save_plotly_fig(fig, path)
    df_data_m = DataFrame(stats_d['mean_l_l'], columns=stats_d['x'], index=stats_d['names'])
    df_ci = DataFrame(stats_d['conf_95'], columns=stats_d['x'], index=stats_d['names'])
    df_mean = DataFrame([stats_d['mean'], stats_d['mean_conf']], columns=stats_d['x'], index=['mean', 'conf_95'])
    df_data_m.to_csv(path + '-data_means.csv')
    df_ci.to_csv(path + '-data_ci.csv')
    df_mean.to_csv(path + '-mean_and_CI.csv')


    #------------------------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------


    names_l = ['amplitude', 'size', 'duration']
    measure_l = ['dffMax2', 'area', 'time_s' ]

    names_l = ['Event number (per minute)', 'amplitude', 'size', 'duration']
    measure_l = [None, 'dffMax2', 'area', 'time_s']
    bh_list_pairs = [['rest', 'running'], ['rest', 'stick_rest'], ['running', 'stick_run_ind_15']]
    bh_list_pairs_names = ['rest_run', 'rest_rest_stick', 'run_run_stick']

    for j, bh_list_pair in enumerate(bh_list_pairs):
        for i, measure in enumerate(measure_l):
            plot_path = os.path.join(output_experiment_path_all_comparison, 'plots', 'transition_dots_{}'.format(bh_list_pairs_names[j]), '{}'.format('dots_'+names_l[i]))
            if 'stick_rest' in bh_list_pair:
                plot, stats_d = astro_plotter.get_measure_all_dot_plot(astroA_l_filt, measure, bh_list=bh_list_pair)
            else:
                plot, stats_d = astro_plotter.get_measure_all_dot_plot(astroA_l, measure, bh_list=bh_list_pair)
            saving_utils.save_plotly_fig(plot, plot_path)

            with open(os.path.join(plot_path + '-data.csv'), mode='w') as csv_file:
                writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

                stats_d['names']

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

            with open(os.path.join(plot_path + '.csv'), mode='w') as csv_file:
                writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

                writer.writerow('')
                writer.writerow(['mean_0', 'mean_1', 'mean_conf_0', 'mean_conf_1'])
                l = []
                l.extend(stats_d['mean'])
                l.extend(stats_d['mean_conf'])
                writer.writerow(l)

    #------------------------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------
    

    path = os.path.join(output_experiment_path_all_comparison, 'plots', 'pdf_norm_fit')

    estimates_d = {}
    all_event_values = {}
    for measure in ['dffMax2' , 'time_s']:
        if measure == 'dffMax2':
            num_bins = 200
            max_filter_val = 3
        elif measure == 'time_s':
            num_bins = 30
            max_filter_val = 2.91
            
        estimates_d[measure] = {}
        all_event_values[measure] = {}
        for bh in ['rest', 'running']:
            fig, x, y_l, all_events_measure_l = astro_plotter.measure_distribution_plot(astroA_l, bh, measure, num_bins=10, min_measure=None, max_measure=None, measure_name=aqua_utils.get_measure_names([measure]), mode='MOE', with_measure_values=True)
            all_events_measure_l = np.array(all_events_measure_l)
            all_events_measure_l = all_events_measure_l[all_events_measure_l < max_filter_val]
            a_estimate, loc_estimate, scale_estimate = skewnorm.fit(all_events_measure_l)
            
            x = np.linspace(np.min(all_events_measure_l), np.max(all_events_measure_l), 100)
            p = skewnorm.pdf(x, a_estimate, loc_estimate, scale_estimate)
            estimates_d[measure][bh] = [a_estimate, loc_estimate, scale_estimate, np.min(x), np.max(x)]
            all_event_values[measure][bh] = np.copy(np.array(all_events_measure_l))
            fig = plotly_utils.plot_scatter_histogram(x=x, y_hist=all_events_measure_l, y_scatter=p, num_bins=num_bins)
            mean, var, skew, kurt = skewnorm.stats(a=a_estimate, loc=loc_estimate, scale=scale_estimate, moments='mvsk')
                
            a, b = np.histogram(all_events_measure_l, bins=num_bins, range=(0, np.max(x)), density=True)
            
            id_ = measure + '_' + bh
            temp_d = {}
            temp_d['Parameters'] = ["a={}".format(a_estimate), "loc={}".format(loc_estimate), "scale={}".format(scale_estimate)]
            temp_d['Properties'] = ["MEAN={}".format(mean), "VAR={}".format(var), "SKEW={}".format(skew),"KURT={}".format(kurt)]
            #print(temp_d)
            saving_utils.save_csv_dict(temp_d, os.path.join(path, id_ + '.csv'), key_order=['Parameters', 'Properties'])
            saving_utils.save_plotly_fig(fig, os.path.join(path, id_))
            #print('skewnorm.pdf(x, a) = 2 * norm.pdf(x) * norm.cdf(a*x)')
            #print('skewnorm.pdf(x, a, loc, scale) is identically equivalent to skewnorm.pdf(y, a) / scale with y = (x - loc) / scale') 

    with_values = True
    for measure in ['dffMax2', 'time_s']:
        est_rest = estimates_d[measure]['rest']
        est_running = estimates_d[measure]['running']
        
        if measure == 'dffMax2':
            x_min = 0.6
            x_max = 3
            nbins = 100
        elif measure == 'time_s':
            x_min = 0
            x_max = 2.91
        else:
            raise NotImplementedError()
        x = np.linspace(x_min, x_max, 500)
        
        if measure == 'duration' or measure == 'time_s':
            tempset = set(list(all_event_values[measure]['rest'])).union(set(list(all_event_values[measure]['running']))) 
            tempset.add(0)
            x_val_bins = np.sort(np.array(list(tempset)))
            x_val_bins = x_val_bins[x_val_bins <= x_max]
            x_val_bins = x_val_bins[x_val_bins >= x_min]
        else:
            x_val_bins = np.linspace(x_min, x_max, nbins)
        
        #Add bin size / 2 to align better
        x_val_diff = 0
        if measure == 'duration' or measure == 'time_s':
            x_val_diff = (x_val_bins[1] - x_val_bins[0]) / 2
        
        p_rest = skewnorm.pdf(x, est_rest[0], est_rest[1], est_rest[2])
        p_running = skewnorm.pdf(x, est_running[0], est_running[1], est_running[2])
        
        if with_values:
            vals_running, vals_x_running = np.histogram(all_event_values[measure]['running'][all_event_values[measure]['running'] < x_max], bins=x_val_bins, density=True)
            vals_rest, vals_x_rest = np.histogram(all_event_values[measure]['rest'][all_event_values[measure]['rest'] < x_max], bins=x_val_bins, density=True)
            
            #Shift by 1 so they look more aligned(due to large bin sizes)
            #e.g. value at 0 is values between 0-(0+bin_size)
            #We are essentially moving the point of values lets say [0, 1] to 0 and then with diff to 0.5
            vals_running = vals_running[1:]
            vals_rest = vals_rest[1:]
            measure_name = aqua_utils.get_measure_names([measure])
            fig = plotly_utils.plot_scatter_mult(x_l=[x, x, vals_x_rest + x_val_diff, vals_x_running + x_val_diff], y_l_l=[p_rest, p_running, vals_rest, vals_running], mode_l=['lines','lines', 'markers','markers'], name_l=['rest','running', 'rest-true', 'running-true'], confidence=False, with_stats=False, title='Skewed distribution: {}'.format(measure_name), x_title=measure_name, y_title='p(X)')
        
        else:
            measure_name = aqua_utils.get_measure_names([measure])
            fig = plotly_utils.plot_scatter_mult(x_l=[x, x], y_l_l=[p_rest, p_running], name_l=['rest','running'], confidence=False, with_stats=False, title='Skewed distribution: {}'.format(measure_name), x_title=measure_name, y_title='p(X)')
        
        id_ = 'measure={}-withvalues={}'.format(measure_name, with_values)
        saving_utils.save_plotly_fig(fig, os.path.join(path, id_))


def generate_axon_plots(axon_plotter, AA_l, output_folder):
    print('---TRANSITION PROPORTION DELAYS PLOT ALL---')
    
    output_experiment_path_all_comparison = os.path.join(output_folder, 'axon_all')
       
    delay_ranges_pairs = [[3*AA_l[0].fr, 6*AA_l[0].fr], [2*AA_l[0].fr, 4*AA_l[0].fr]]
    delay_ranges_pairs = [[int(v[0]), int(v[1])] for v in delay_ranges_pairs]
    before_range_3, after_range_6 = delay_ranges_pairs[0]
    before_range_2, after_range_4 = delay_ranges_pairs[1]

    print('Alt Proportion plots...')
    

    rest_to_run_setting = { 
        'before_bh':'rest_semi_exact', 
        'inds_bh':'running_exact_start', 
        'after_bh':'running_semi_exact', 
        'before_range' : before_range_3,
        'after_range' : after_range_6,
        'fit': True, 
        'delay_step_size': 10,
        'confidence': True}


    __save_axon_transition_plot(axon_plotter=axon_plotter, 
                    AA_l=AA_l, 
                    setting=rest_to_run_setting, 
                    plot_type='behaviour', 
                    path=os.path.join(output_experiment_path_all_comparison, 'plots', f'rest_to_run_speed'), 
                    bh_measure='speed')

    __save_axon_transition_plot(axon_plotter=axon_plotter, 
                AA_l=AA_l, 
                setting=rest_to_run_setting, 
                plot_type='proportions_stick_filter', 
                path=os.path.join(output_experiment_path_all_comparison, 'plots', f'rest_to_run_vibrisastimtiming'), 
                bh_measure=None)

    for aa_setting in ['axon']:
        rest_to_run_setting['aa_setting'] = aa_setting

        __save_axon_transition_plot(axon_plotter=axon_plotter, 
                                    AA_l=AA_l, 
                                    setting=rest_to_run_setting, 
                                    plot_type='proportions', 
                                    path=os.path.join(output_experiment_path_all_comparison, 'plots', f'rest_to_run_{aa_setting}_proportions'), 
                                    bh_measure=None)
    

    run_to_rest_setting = { 
        'before_bh':'running_semi_exact', 
        'inds_bh':'rest_start', 
        'after_bh':'rest_semi_exact', 
        'before_range' : before_range_3,
        'after_range' : after_range_6,
        'fit': True, 
        'delay_step_size': 10,
        'confidence': True
    }

    __save_axon_transition_plot(axon_plotter=axon_plotter, 
                AA_l=AA_l, 
                setting=run_to_rest_setting, 
                plot_type='behaviour', 
                path=os.path.join(output_experiment_path_all_comparison, 'plots', f'run_to_rest_speed'), 
                bh_measure='speed')

    for aa_setting in ['axon']:
        run_to_rest_setting['aa_setting'] = aa_setting

        __save_axon_transition_plot(axon_plotter=axon_plotter, 
                                    AA_l=AA_l, 
                                    setting=run_to_rest_setting, 
                                    plot_type='proportions', 
                                    path=os.path.join(output_experiment_path_all_comparison, 'plots', f'run_to_rest_{aa_setting}_proportions'), 
                                    bh_measure=None)

    
    run_stick_run_setting = { 
        'before_bh':'running_semi_exact', 
        'inds_bh':'stick_exact_start', 
        'after_bh':'running_semi_exact', 
        'before_range' : before_range_2,
        'after_range' : after_range_4,
        'fit': True, 
        'delay_step_size': 10,
        'confidence': True
    }

    __save_axon_transition_plot(axon_plotter=axon_plotter, 
                AA_l=AA_l, 
                setting=run_stick_run_setting, 
                plot_type='behaviour', 
                path=os.path.join(output_experiment_path_all_comparison, 'plots', f'run_stick_run_speed'), 
                bh_measure='speed')

    __save_axon_transition_plot(axon_plotter=axon_plotter, 
                AA_l=AA_l, 
                setting=run_stick_run_setting, 
                plot_type='proportions_stick_filter', 
                path=os.path.join(output_experiment_path_all_comparison, 'plots', f'run_stick_run_vibrisastimtiming'), 
                bh_measure=None)

    
    for aa_setting in ['axon', 'astro']:
        run_stick_run_setting['aa_setting'] = aa_setting

        __save_axon_transition_plot(axon_plotter=axon_plotter, 
                                    AA_l=AA_l, 
                                    setting=run_stick_run_setting, 
                                    plot_type='proportions', 
                                    path=os.path.join(output_experiment_path_all_comparison, 'plots', f'run_stick_run_{aa_setting}_proportions'), 
                                    bh_measure=None)

def __save_astro_transition_plot(astro_plotter, astroA_l, setting, plot_type, path, measure=None, bh_measure=None):
    measure_y_titles = {'dffMax2default' : 'Amplitude', 
                        'time_s' : 'Duration (s)', 
                        'area' : 'Size'}
    bh_measure_y_titles = {'speed' : 'Speed (cm/s)'}

    before_bh=setting['before_bh']
    inds_bh = setting['inds_bh']
    after_bh = setting['after_bh']

    before_range = setting['before_range']
    after_range = setting['after_range']
    fit = setting['fit']
    delay_step_size = setting['delay_step_size']
    confidence = setting['confidence']

    p = {'fit' : fit, 'delay_step_size' : delay_step_size, 'confidence' : confidence}

    if plot_type == 'proportions':
        fig_d, bin_stats = astro_plotter.get_transition_proportion_delays_plot_all_alt(astroA_l, 
                                                                    before_bh=before_bh, inds_bh=inds_bh, after_bh=after_bh,
                                                                    before_range=before_range, after_range=after_range,
                                                                    **p)
    elif plot_type == 'measure':
        assert measure is not None
        fig_d, bin_stats = astro_plotter.get_transition_proportion_delays_plot_all_alt(astroA_l, before_bh=before_bh, inds_bh=inds_bh, after_bh=after_bh,
                                                                                before_range=before_range, after_range=after_range,
                                                                                measure=measure,
                                                                                y_title=measure_y_titles[measure],
                                                                                **p)

    elif plot_type == 'behaviour':
        assert bh_measure is not None
        fig_d, bin_stats = astro_plotter.get_transition_bh_values_plot_all_alt(astroA_l, 
                                                            before_bh=before_bh, inds_bh=inds_bh, after_bh=after_bh,
                                                            bh_measure=bh_measure,
                                                            before_range=before_range, after_range=after_range,
                                                            y_title=bh_measure_y_titles[bh_measure],
                                                            **p)
    else:
        raise ValueError('Plot type must be "proportions", "measure"')

    fig_v = fig_d['event_avg_no_mult']
    fig_id = os.path.join(path, 'range_{}_{}-step_{}'.format(before_range, after_range, delay_step_size))
    saving_utils.save_plotly_fig(fig_v, fig_id)
    saving_utils.save_csv_dict(bin_stats, path=fig_id + '.csv', key_order=['x', 'mean', 'std', 'confidence_95'])

    all_data_dict = {bin_stats['x'][i]:bin_stats['y_all'][:, i] for i in range(len(bin_stats['x']))}

    saving_utils.dict_to_csv(all_data_dict, name=fig_id + 'range_{}_{}-step_{}-data.csv'.format(before_range, after_range, delay_step_size), base_folder=path)
    #DataFrame(bin_stats['y_all'], columns=bin_stats['x']).to_csv(data_csv_path, index=False)
    
def __save_axon_transition_plot(axon_plotter, AA_l, setting, plot_type, path, bh_measure=None):
    bh_measure_y_titles = {'speed' : 'Speed (cm/s)'}
    
    before_bh = setting['before_bh']
    inds_bh = setting['inds_bh']
    after_bh =  setting['after_bh']

    before_range = setting['before_range']
    after_range = setting['after_range']
    fit = setting['fit']
    delay_step_size = setting['delay_step_size']
    confidence = setting['confidence']

    if 'aa_setting' in setting:
        aa_setting = setting['aa_setting']
        p = {'fit' : fit, 'delay_step_size' : delay_step_size, 'confidence' : confidence, 'setting' : aa_setting}
    else:
        p = {'fit' : fit, 'delay_step_size' : delay_step_size, 'confidence' : confidence}

    if plot_type == 'proportions':
        fig_d, bin_stats = axon_plotter.get_axon_transition_proportion_delays_plot_all(AA_l, before_bh=before_bh, inds_bh=inds_bh, after_bh=after_bh,
                                                                                    before_range=before_range, after_range=after_range,
                                                                                    **p)
    elif plot_type == 'behaviour':
        assert bh_measure is not None
        fig_d, bin_stats = axon_plotter.get_transition_bh_values_plot_all_alt(AA_l,
                                                                before_bh=before_bh, inds_bh=inds_bh, after_bh=after_bh,
                                                                bh_measure=bh_measure,
                                                                before_range=before_range, after_range=after_range,
                                                                y_title=bh_measure_y_titles[bh_measure],
                                                                **p)
    elif plot_type == 'proportions_stick_filter':
        fig_d, bin_stats = axon_plotter.get_axon_transition_proportion_delays_STICK_FILTER_plot_all(AA_l, before_bh=before_bh, inds_bh=inds_bh, after_bh=after_bh,
                                                                                before_range=before_range, after_range=after_range,
                                                                                **p)
    else:
        raise ValueError('Invalid plot type')

    fig_v = fig_d['event_avg_no_mult']
    fig_id = os.path.join(path, 'range_{}_{}-step_{}'.format(before_range, after_range, delay_step_size))
    saving_utils.save_plotly_fig(fig_v, fig_id)
    saving_utils.save_csv_dict(bin_stats, path=fig_id + '.csv', key_order=['x', 'mean', 'std', 'confidence_95'])