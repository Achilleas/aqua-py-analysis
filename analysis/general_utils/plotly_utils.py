import plotly.graph_objs as go
from plotly import tools
import numpy as np
import scipy.interpolate as interp
from analysis.general_utils import stat_utils, general_utils
from scipy import optimize
import seaborn as sns
import matplotlib.pyplot as plt

def plot_2D_graph(arr, title='Parametric Plot'):
    x = np.arange(0, arr.shape[0])
    y = np.arange(0, arr.shape[1])

    yGrid, xGrid = np.meshgrid(x, y)

    surface = go.Surface(x=xGrid, y=yGrid, z=arr)

    data = [surface]

    layout = go.Layout(
        title=title,
        scene=dict(
            xaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)',
                autorange='reversed'
            ),
            yaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            ),
            zaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            )
        )
    )

    fig = go.Figure(data=data, layout=layout)
    return fig

def plot_waterfall(arrays_l, legend_names, title='delays', x_title='', y_title=''):
    traces_l = []

    for i in range(len(arrays_l)):
        trace_i = go.Scatter(
            x = arrays_l[i],
            y = np.arange(len(arrays_l[i]), 0, -1),
            mode = 'markers',
            name = legend_names[i],
            opacity = 0.5
        )

        traces_l.append(trace_i)

    layout = go.Layout(title=title, xaxis=dict(title=x_title), yaxis=dict(title=y_title),)
    fig = go.Figure(data=traces_l, layout=layout)
    return fig

def plot_waterfall_interpolate(arrays_l, legend_names, title='delays', x_title='', y_title=''):
    traces_l = []
    max_size = np.max([len(arr) for arr in arrays_l])

    for i in range(len(arrays_l)):
        array_i = arrays_l[i]
        array_i_interp = interp.interp1d(np.arange(len(array_i)),array_i)
        array_i_stretch = array_i_interp(np.linspace(0,len(array_i)-1, max_size))
        array_i_floor = np.floor(array_i_stretch)

        trace_i = go.Scatter(
            x = array_i_floor,
            y = np.arange(len(array_i_floor), 0, -1),
            mode = 'markers',
            name = legend_names[i],
            opacity = 0.5
        )
        traces_l.append(trace_i)
    layout = go.Layout(title=title, xaxis=dict(title=x_title), yaxis=dict(title=y_title),)
    fig = go.Figure(data=traces_l, layout=layout)
    return fig

def plot_bar(x, y, text_values=None, title='', text_size=10, x_title='', y_title='', margin_l=100, margin_b=100, err_y=[], err_symmetric=True):
    err_y_visible=True
    if len(err_y) == 0:
        err_y_visible = False

    bar = go.Bar(
        x = x,
        y = y,
        text=text_values,
        name = "count",
        textposition = 'auto',
        marker=dict(
        color='rgba(0,76,153, 0.7)',
        line=dict(
                color='rgba(55, 128, 191, 1.0)',
                width=2,
            )
        ),
        error_y=dict(
            type='data',
            array=err_y,
            visible=True,
            symmetric=err_symmetric
        ),
        textfont=dict(size=text_size)
    ),

    layout = go.Layout(
        title=title,
        xaxis=dict(title=x_title),
        yaxis=dict(title=y_title),
        margin=dict(
            l=margin_l,
            r=50,
            b=margin_b,
            t=100,
        ),
    )
    fig = go.Figure(data=bar, layout=layout)
    return fig

def plot_group_bar(x, y_l, text_values_l=None, title='', text_size=10,
                    x_title='', y_title='', legends=[], err_y=[], std_y=[],
                    margin_l=100, margin_b=100, margin_r=100):
    err_y_visible=True
    if len(err_y) == 0:
        err_y_visible = False
    data = []

    if len(legends) == 0:
        legends = ['']*len(y_l)

    if text_values_l is None:
        text_values_l = [None]*len(y_l)

    for i, y in enumerate(y_l):
        trace = go.Bar(
            x = x,
            y = y,
            text=text_values_l[i],
            name = legends[i],
            textposition = 'inside',
            marker=dict(
            #color='rgba(0,76,153, 0.7)',
            line=dict(
                    #color='rgba(55, 128, 191, 1.0)',
                    #width=2,
                )
            ),
            error_y=dict(
                type='data',
                array=std_y[i],
                visible=True
            ),
            textfont=dict(size=text_size)
        )
        data.append(trace)
    layout = go.Layout(title=title,
                        xaxis=dict(title=x_title),
                        yaxis=dict(title=y_title),
                        barmode='group',
                        margin=dict(
                            l=margin_l,
                            r=margin_r,
                            b=margin_b,
                            t=100,
                        ),)
    fig = go.Figure(data=data, layout=layout)
    return fig

def plot_histogram(arr, title='histogram', x_title='', y_title=''):
    # Plot histogram of signal lenghts over whole thing
    histogram = [go.Histogram(x=arr, histnorm='probability')]
    layout = go.Layout(title=title, xaxis=dict(title=x_title),
                        yaxis=dict(title=y_title),)
    fig = go.Figure(data=histogram, layout=layout)
    return fig

def plot_scatter(x, y , mode='lines', title='scatter', x_title='', y_title='', bin_size=1, bin_type='mean'):
    if bin_size > 1:
        y = bin_avg_arr(y, bin_size, bin_type=bin_type)

        if len(x) != len(y):
            x = np.arange(np.min(x), np.max(x)+1-bin_size, bin_size)

    trace0 = go.Scatter(
        x=x,
        y=y,
        mode=mode
    )

    layout = go.Layout(title=title, xaxis=dict(title=x_title),
                        yaxis=dict(title=y_title),)
    fig = go.Figure(data=[trace0], layout=layout)
    return fig

def plot_scatter_error(x, y, mode='lines', title='scatter', x_title='', y_title='', with_points=True, with_error=True, exp_fit=False, with_details=False):
    means_y = np.array([np.mean(y_v) for y_v in y])
    confidence_low = [stat_utils.mean_confidence_interval(y_v, confidence=0.95)[1] for y_v in y]
    confidence_high = [stat_utils.mean_confidence_interval(y_v, confidence=0.95)[2] for y_v in y]

    conf_95 = [confidence_high[i] - means_y[i] for i in range(len(means_y))]


    trace = go.Scatter(
        name='Measurement',
        x=x,
        y=means_y,
        mode=mode,
        line=dict(color='rgb(31, 119, 180)'))

    data = []
    if not with_error:
        data = [trace]

    if with_error:
        upper_bound = go.Scatter(
            name='Upper/Lower Bound',
            x=x,
            y=confidence_high,
            mode=mode,
            marker=dict(color="#444"),
            line=dict(width=1, color='rgb(66,167,244)', dash='longdash'))

        lower_bound = go.Scatter(
            x=x,
            y=confidence_low,
            mode=mode,
            marker=dict(color="#444"),
            line=dict(width=1, color='rgb(66,167,244)', dash='longdash'),
            showlegend=False)
        data.append(lower_bound)
        data.append(upper_bound)
        data.append(trace)

    y_fit = None
    if exp_fit:
        def test_func(x, N, b):
            return N*(np.exp(-(x/b)))
        params, params_covariance = optimize.curve_fit(test_func, x, means_y)
        y_fit = test_func(x, *params)
        par = [v for v in params]
        print(par)
        fit_scatter = go.Scatter(
            x=x,
            y=y_fit,
            mode=mode,
            marker=dict(color='rgb(244,143,66)'),
            name='{:.1e}*exp<sup>-(t/{:.1e})<sup>'.format(*par)
        )

        data.append(fit_scatter)

    if with_points:
        traces_individual_l = []
        for i in range(len(y[0])):
            traces_individual_l.append(go.Scatter(
                x=x,
                y=[y_v[i] for y_v in y],
                mode='markers',
                opacity=0.6,
                showlegend=False
            )
        )
        data.extend(traces_individual_l)


    layout = go.Layout(title=title, xaxis=dict(title=x_title),
                        yaxis=dict(title=y_title),)
    fig = go.Figure(data=data, layout=layout)

    if with_details:
        return fig, {'x' : x, 'data' : y, 'mean' : means_y, 'conf_95' : conf_95, 'fit' : y_fit}
    return fig

def plot_scatter_fmt(x, y , mode='lines', title='scatter', astype='float', straight_lines_only=True, x_title='', y_title=''):
    '''
    If straight lines only make sure x values are evenly spaced out, otherwize not implemented
    '''
    if straight_lines_only:
        new_x = []
        new_y = []

        new_x.append(x[0])
        new_y.append(y[0])
        for x_i in range(1, len(x)):
            x_prev = x[x_i - 1]
            x_next = x[x_i]
            x_mid = (x_next + x_prev) / 2.0

            new_x.append(x_mid)
            new_x.append(x_mid)
            new_x.append(x_next)

            new_y.append(y[x_i-1])
            new_y.append(y[x_i])
            new_y.append(y[x_i])
        trace0 = go.Scatter(
            x=np.array(new_x),
            y=np.array(new_y).astype(astype),
            mode=mode
        )
    else:
        trace0 = go.Scatter(
            x=x,
            y=y.astype(int),
            mode=mode
        )

    layout = go.Layout(title=title, xaxis=dict(title=x_title),
                        yaxis=dict(title=y_title, nticks=2),)
    fig = go.Figure(data=[trace0], layout=layout)
    return fig

def plot_scatter_mult(x_l, y_l_l, name_l, mode='lines', title='scatter', x_title='', y_title='',
                        xrange=None, yrange=None, confidence=True, with_stats=True, point_box=False):
    """
    In plot_scatter_mult we pass y_l, a list of values of y. Here we pass a list of lists of values of y
    For each datapoint in list we calculate the 95% confidence intervals

    y_l_l contains an array of arrays of arrays (to allow for multiple lines and multiple confidence intervals)
    """

    colour_l = ['rgb(66,244,143)',
                'rgb(66,167,244)',
                'rgb(244,66,167)',
                'rgb(244,143,66)',
                'rgb(244,66,78)',
                'rgb(244,232,66)',
                'rgb(10,10,10)',
                'rgb(255, 102, 102)',
                'rgb(204, 0, 204)',
                'rgb(51, 153, 255)',
                'rgb(153, 0, 76)',
                'rgb(76, 153, 0)',
                'rgb(0, 76, 153)',
                'rgb(153, 153, 153)',
                'rgb(0, 255, 128)',
                'rgb(204, 153, 255)',
                'rgb(204, 204, 0)',
                'rgb(0, 102, 102)']

    traces_l = []

    conf_low_l = []
    conf_high_l = []
    conf = []
    std = []
    mean_l = []

    mean_l_l = []
    conf_low_l_l = []
    conf_high_l_l = []

    for y_i, y_l in enumerate(y_l_l):
        mean_l_l.append([stat_utils.mean_confidence_interval(y_v, confidence=0.95)[0] for y_v in y_l])

        if with_stats or confidence:
            conf_low_l_l.append([stat_utils.mean_confidence_interval(y_v, confidence=0.95)[1] for y_v in y_l])
            conf_high_l_l.append([stat_utils.mean_confidence_interval(y_v, confidence=0.95)[2] for y_v in y_l])

            conf.append([stat_utils.mean_confidence_interval(y_v, confidence=0.95)[0] - \
                         stat_utils.mean_confidence_interval(y_v, confidence=0.95)[1] for y_v in y_l])
            std.append([np.std(y_v) for y_v in y_l])

    for i in range(len(x_l)):
        trace_i = go.Scatter(
            x=x_l[i],
            y=mean_l_l[i],
            mode=mode,
            name=name_l[i],
            line=dict(color=colour_l[i])
        )
        if confidence:
            upper_bound = go.Scatter(
                x=x_l[i],
                y=conf_high_l_l[i],
                mode=mode,
                marker=dict(color="#444"),
                line=dict(width=1, color=colour_l[i], dash='longdash'),
                showlegend=False
                )

            lower_bound = go.Scatter(
                x=x_l[i],
                y=conf_low_l_l[i],
                mode=mode,
                marker=dict(color="#444"),
                line=dict(width=1, color=colour_l[i], dash='longdash'),
                showlegend=False,
                )
            traces_l.append(lower_bound)
            traces_l.append(upper_bound)
        traces_l.append(trace_i)

    layout = go.Layout(title=title, xaxis=dict(title=x_title),
                        yaxis=dict(title=y_title),)

    if yrange is not None:
        layout.update(yaxis=dict(range=yrange))
    if xrange is not None:
        layout.update(xaxis=dict(range=xrange))

    fig = go.Figure(data=traces_l, layout=layout)

    if with_stats and not confidence:
        return fig, {'x' : x_l, 'data' : y_l_l, 'mean' : mean_l_l, 'names' : name_l}
    if with_stats and confidence:
        return fig, {'x' : x_l, 'data' : y_l_l, 'conf_95' : conf, 'std' : std, 'mean' : mean_l_l, 'names' : name_l}
    return fig


def plot_scatter_mult_with_avg(x_l, y_l_l, y_mean, name_l, mode='lines', title='scatter', x_title='', y_title='',
                        xrange=None, yrange=None, confidence=True, avg_confidence=True, with_stats=True, point_box=False, mean_width_size=10):
    """
    In plot_scatter_mult we pass y_l, a list of values of y. Here we pass a list of lists of values of y
    For each datapoint in list we calculate the 95% confidence intervals

    y_l_l contains an array of arrays of arrays (to allow for multiple lines and multiple confidence intervals)
    """

    traces_l = []

    conf_low_l = []
    conf_high_l = []
    conf = []
    std = []
    mean_l = []

    mean_l_l = []
    conf_low_l_l = []
    conf_high_l_l = []

    for y_l in y_l_l:
        print('LENS?', [len(y_v) for y_v in y_l])
        mean_l_l.append([stat_utils.mean_confidence_interval(y_v, confidence=0.95)[0] for y_v in y_l])

        if with_stats or confidence:
            conf_low_l_l.append([stat_utils.mean_confidence_interval(y_v, confidence=0.95)[1] for y_v in y_l])
            conf_high_l_l.append([stat_utils.mean_confidence_interval(y_v, confidence=0.95)[2] for y_v in y_l])

            conf.append([stat_utils.mean_confidence_interval(y_v, confidence=0.95)[0] - \
                         stat_utils.mean_confidence_interval(y_v, confidence=0.95)[1] for y_v in y_l])
            std.append([np.std(y_v) for y_v in y_l])

    colour_l = ['rgb(66,244,143)',
                'rgb(66,167,244)',
                'rgb(244,66,167)',
                'rgb(244,143,66)',
                'rgb(244,66,78)',
                'rgb(244,232,66)',
                'rgb(10,10,10)',
                'rgb(255, 102, 102)',
                'rgb(204, 0, 204)',
                'rgb(51, 153, 255)',
                'rgb(153, 0, 76)',
                'rgb(76, 153, 0)',
                'rgb(0, 76, 153)',
                'rgb(153, 153, 153)',
                'rgb(0, 255, 128)',
                'rgb(204, 153, 255)',
                'rgb(204, 204, 0)',
                'rgb(0, 102, 102)']

    traces_l = []
    shapes_l = []

    for i in range(len(y_l_l)):
        trace_i = go.Scatter(
            x=x_l,
            y=mean_l_l[i],
            mode='lines+markers',
            line=dict(color=colour_l[i]),
            marker=dict(size=15),
            error_y=dict(
                type='data',
                symmetric=True,
                array=np.array(conf_high_l_l[i]) - np.array(mean_l_l[i]),
                color=colour_l[i]
            ),
            opacity=0.5,
            name=name_l[i]
        )
        traces_l.append(trace_i)


    error_y=None
    if confidence or avg_confidence:
        mean_np_l_l = np.array(mean_l_l)
        mean_l = []
        conf_l = []
        for i in range(mean_np_l_l.shape[1]):
             mean, conf_low, conf_high = stat_utils.mean_confidence_interval(mean_np_l_l[:, i], confidence=0.95)
             mean_l.append(mean)
             conf_l.append(conf_high-mean)
        error_y=dict(
            type='data',
            symmetric=True,
            array=conf_l,
            color=colour_l[10],
        )

    if y_mean != None:
        mean_trace = go.Scatter(
            x=x_l,
            y=y_mean,
            mode='lines',
            name='Mean',
            line=dict(color=colour_l[10]),
            error_y=error_y
        )
    else:
        mean_trace = go.Scatter(
            x=x_l,
            y=np.nanmean(np.array(mean_l_l), axis=0),
            mode='lines',
            name='Mean',
            line=dict(color=colour_l[10], width=mean_width_size),
            error_y=error_y
        )
    traces_l.append(mean_trace)
    layout = go.Layout( title=title,
                        xaxis=dict(title=x_title),
                        yaxis=dict(title=y_title),
                        shapes=shapes_l
                        )
    if yrange is not None:
        layout.update(yaxis=dict(range=yrange))
    if xrange is not None:
        layout.update(xaxis=dict(range=xrange))

    fig = go.Figure(data=traces_l, layout=layout)

    if with_stats and (not avg_confidence) and (not confidence):
        return fig, {'x' : x_l, 'data' : y_l_l, 'mean_l_l' : mean_l_l, 'mean' : mean_l if y_mean is None else y_mean, 'names' : name_l}
    if (with_stats) and (avg_confidence) and (not confidence):
        return fig, {'x' : x_l, 'data' : y_l_l, 'mean_l_l' : mean_l_l, 'mean_conf' : conf_l, 'mean' : mean_l if y_mean is None else y_mean, 'names' : name_l}
    if with_stats and confidence:
        return fig, {'x' : x_l, 'data' : y_l_l, 'conf_95' : conf, 'std' : std, 'mean_l_l' : mean_l_l, 'mean_conf' : conf_l, 'mean' : mean_l if y_mean is None else y_mean, 'names' : name_l}
    return fig

def plot_scatter_mult_tree(x, y_main, y_mult, mode_main='lines', mode_mult='markers',
                            title='', y_title='', x_title='', fit=False, fit_annotation_pos_fix=1,
                            bin_main_size=1, bin_mult_size=1, opacity=0.1, confidence=False, with_stats=False,
                            y_mult_include=True, confidence_format='lines', bin_type='mean'):
    print('BIN TYPE', bin_type)
    '''
    x = [len_x]
    y_main = [len_x]
    y_mult = [n, len_x]
    '''
    traces_l = []

    if bin_mult_size > 1:
        #Split in bins:
        remove_end = y_mult.shape[1] % bin_mult_size

        mult_bins = np.split(y_mult[:, :-remove_end], bin_mult_size, axis=1)
        #(273, 61)

        x_mult_avg = np.arange(np.min(x), np.max(x)+1-bin_mult_size, bin_mult_size)
        y_mult_bins = np.zeros([y_mult.shape[0]*bin_mult_size, len(x_mult_avg)])

        for i in range(len(x_mult_avg)):
            y_mult_bins[:, i] = (y_mult[:, i*bin_mult_size:(i+1)*bin_mult_size]).flatten()

        y_mult = np.copy(y_mult_bins)
    #Confidence intervals
    #---------------------------------------------
    if confidence or with_stats:
        mean_l = []
        conf_low_l = []
        conf_high_l = []

        confidence_l = []
        std_l = []

        for i in range(y_mult.shape[1]):
            r = y_mult[:, i]
            valid_r = r[~np.isnan(r)]

            if bin_type == 'add':
                valid_r *= bin_mult_size

            mean, conf_low, conf_high = stat_utils.mean_confidence_interval(valid_r, confidence=0.95)
            mean_l.append(mean)
            conf_low_l.append(conf_low)
            conf_high_l.append(conf_high)

            confidence_l.append(conf_high - mean)
            std_l.append(np.std(valid_r))


        #Just make a bar plot if we have 2 values to compare before and after
        if len(mean_l) == 2:
            #TODO
            #Hard coded
            fig = plot_bar(x=['before', 'after'], y=mean_l, err_y=confidence_l, title=title, y_title=y_title, x_title=x_title)

            if with_stats:
                return fig, {'confidence_95' : confidence_l, 'std' : std_l, 'mean' : mean_l, 'x' : (x if bin_mult_size == 1 else x_mult_avg), 'y_all' : y_mult}
            else:
                return fig

        if len(mean_l) == 1:
            raise NotImplementedError
    #---------------------------------------------

    if bin_main_size > 1:
        y_main_avg = bin_avg_arr(y_main, bin_main_size, bin_type=bin_type)
        if confidence:
            y_main_avg = np.array(mean_l)
        x_avg = np.arange(np.min(x), np.max(x)+1-bin_main_size, bin_main_size)

    if fit:
        def test_func(x, a, b, c, d):
            return a + b*x + c*(x**2) + d*(x**3)

        if len(x) // bin_main_size > 3:
            if bin_main_size > 1:
                params, params_covariance = optimize.curve_fit(test_func, x_avg, y_main_avg)
            else:
                params, params_covariance = optimize.curve_fit(test_func, x, y_main)
            y_fit = test_func(x, *params)
        else:
            fit = False

    colour_main = 'rgb(66,167,244)'
    colour_main_avg = 'rgb(48,158,45)'
    colour_mult = 'rgb(66,244,143)'
    colour_fit = 'rgb(244,66,167)'
    colour_conf = '#444'

    shapes = None
    if confidence:
        if confidence_format == 'lines':
            upper_bound = go.Scatter(
                name='Upper/Lower Bound',
                x=x if bin_mult_size == 1 else x_mult_avg,
                y=np.array(conf_high_l),
                mode='lines',
                marker=dict(color=colour_conf),
                line=dict(width=1, color='rgb(66,167,244)', dash='longdash'))

            lower_bound = go.Scatter(
                x=x if bin_mult_size == 1 else x_mult_avg,
                y=np.array(conf_low_l),
                mode='lines',
                marker=dict(color=colour_conf),
                line=dict(width=1, color='rgb(66,167,244)', dash='longdash'),
                showlegend=False)
            print('Lower bound')
            traces_l.append(lower_bound)
            traces_l.append(upper_bound)
        elif confidence_format == 'bar':
            ppb_x = x if bin_mult_size == 1 else x_mult_avg
            ppb_x = ppb_x / fit_annotation_pos_fix
            d = plot_point_box_revised(ppb_x, list(y_mult.transpose(1,0)),
                                        return_details=True, point_size_mult=1, interval_size_mult=1,
                                        ignore_points=True, showlegend=False, squish_y=0.5)
            for trace in d['traces']:
                traces_l.append(trace)
            shapes = d['shapes']

    #Main trace
    traces_l.append(
        go.Scatter(
            x=x,
            y=y_main,
            mode=mode_main,
            line=dict(color=colour_main),
            name='Average'
        )
    )
    print('Added trace 1')

    #Main avg trace
    if bin_main_size > 1:
        print('Added trace 2')
        traces_l.append(
            go.Scatter(
                x=x_avg,
                y=y_main_avg,
                mode=mode_main,
                line=dict(color=colour_main_avg),
                name='Binned average'
            )
        )

    if y_mult_include:
        print(y_mult.shape)
        for i in range(y_mult.shape[0]):
            traces_l.append(
                go.Scatter(
                    x=x if bin_mult_size == 1 else x_mult_avg,
                    y=y_mult[i, :],
                    mode=mode_mult,
                    opacity=opacity,
                    line=dict(color=colour_mult, ),
                    showlegend=False
                )
            )

    if fit:
        traces_l.append(
            go.Scatter(
                x=x,
                y=y_fit,
                mode=mode_main,
                line=dict(color=colour_fit),
                name='Polynomial fit'
            )
        )

    layout = go.Layout(title=title,
                        xaxis=dict(title=x_title),
                        yaxis=dict(title=y_title),
                        shapes=shapes
                        )

    if fit:
        layout.update(
            annotations=[
              dict(
                  x=x[len(x)//2] / fit_annotation_pos_fix,
                  y=y_fit[len(x)//2],
                  xref='x',
                  yref='y',
                  text='{:.1e} + {:.1e}x + {:.1e}x<sup>2</sup> + {:.2e}x**3'.format(*[v for v in params]),
                  showarrow=True,
                  arrowhead=6,
              ),
          ]
        )
    fig = go.Figure(data=traces_l, layout=layout)

    if with_stats:
        print('RETURNING HERE???')
        return fig, {'confidence_95' : confidence_l, 'std' : std_l, 'mean' : mean_l, 'x' : (x if bin_mult_size == 1 else x_mult_avg), 'y_all' : y_mult}
    return fig

def plot_scatter_signal(x, y, begin_i, end_i, mode='lines', title='scatter', x_title='', y_title='', with_legend=False):
    traces_l = []

    x2 = np.arange(begin_i, end_i)
    y2 = y[begin_i:end_i]

    x_l = [x, x2]
    y_l = [y, y2]

    for i in range(len(x_l)):
        trace_i = go.Scatter(
            x=x_l[i],
            y=y_l[i],
            mode=mode
        )
        traces_l.append(trace_i)

    layout = go.Layout(title=title, xaxis=dict(title=x_title),
                        yaxis=dict(title=y_title),
                              annotations=[
                                dict(
                                    x=begin_i,
                                    y=y_l[0][begin_i],
                                    xref='x',
                                    yref='y',
                                    text='{}'.format(begin_i),
                                    showarrow=True,
                                    arrowhead=6,
                                    ax=-40,
                                    ay=-20
                                ),
                                dict(
                                    x=end_i,
                                    y=y_l[0][end_i-1],
                                    xref='x',
                                    yref='y',
                                    text='{}'.format(end_i),
                                    showarrow=True,
                                    arrowhead=6,
                                    ax=40,
                                    ay=-20
                                )],showlegend=False
                      )
    fig = go.Figure(data=traces_l, layout=layout)
    return fig


#TODO
def plot_scatter_signal_mult(x, y_l, begin_l, end_l, mode='lines', title='scatter', x_title='', y_title='', with_legend=False):
    traces_l = []

    x2_l = [np.arange(begin_i, end_i) for (begin_i, end_i) in zip(begin_l, end_l)]
    y2_l = [y_l[i][begin_i:end_i] for i, (begin_i, end_i) in enumerate(zip(begin_l, end_l))]

    print([[begin_i, end_i] for (begin_i, end_i) in zip(begin_l, end_l)])
    #x_l = [x, x2]
    #y_l = [y, y2]

    for i in range(len(y_l)):
        trace_i = go.Scatter(
            x=x,
            y=y_l[i],
            mode=mode
        )
        traces_l.append(trace_i)

        trace_i = go.Scatter(
            x=x2_l[i],
            y=y2_l[i],
            mode=mode
        )

        traces_l.append(trace_i)

    layout = go.Layout(title=title, xaxis=dict(title=x_title),
                        yaxis=dict(title=y_title)
                      )
    fig = go.Figure(data=traces_l, layout=layout)
    return fig


def plot_heatmap(arr, title='heatmap', color_bar_title='', color_scale='Portland', height=400, width=600):
    heatmap = go.Heatmap(z=arr, x=np.arange(arr.shape[0]), y=np.arange(arr.shape[1]),
                            colorscale=color_scale,
                            colorbar=dict(
                                title=color_bar_title,
                                titleside='right',
                                titlefont=dict(
                                    size=14,
                                    family='Arial, sans-serif',
                                ),
                        )
                )
    layout = go.Layout(title=title, height=height, width=width)
    fig = go.Figure(data=[heatmap], layout=layout)
    return fig

def plot_contour(arr, title='contour', color_bar_title='', color_scale='Portland', line_width=0.1, height=400, width=600, num_ticks=5):
    # ['Greys', 'YlGnBu', 'Greens', 'YlOrRd', 'Bluered', 'RdBu',
    #            'Reds', 'Blues', 'Picnic', 'Rainbow', 'Portland', 'Jet',
    #            'Hot', 'Blackbody', 'Earth', 'Electric', 'Viridis', 'Cividis']

    min_v = np.min(arr)
    max_v = np.max(arr)

    tick_x = list(np.arange(min_v, max_v, (max_v-min_v) / num_ticks))
    #print('tick first', tick_x[1])
    #print('log thingy', -np.log10(tick_x[1]))

    if -np.log10(tick_x[1]) > 0:
        print('Truncating ', int(-np.log10(tick_x[1]))+1)
        print([v for v in tick_x])
        tick_x = [general_utils.truncate(v, int(-np.log10(tick_x[1]))+1) for v in tick_x]
    else:
        print('Truncating', int(-np.log10(tick_x[1])))
        tick_x = [general_utils.truncate(v, int(-np.log10(tick_x[1]))) for v in tick_x]

    if tick_x[1] > 10:
        tick_x = [int(v) for v in tick_x]

    #print('min' ,min_v)
    #print('max', max_v)
    #print('num ticks', num_ticks)
    print('TICK X final', tick_x)

    contour = go.Contour(z=arr, x=np.arange(arr.shape[0]), y=np.arange(arr.shape[1]),
                colorscale=color_scale,
                colorbar=dict(
                    title=color_bar_title,
                    titleside='right',
                    titlefont=dict(
                        size=14,
                        family='Arial, sans-serif',
                    ),
                    tickmode='array',
                    tickvals=tick_x,
                    ticktext=tick_x,
                ),
                line=dict(width=line_width, smoothing=0),
            )

    layout = go.Layout(title=title, height=height, width=width)
    fig = go.Figure(data=[contour], layout=layout)
    return fig

def plot_contour_threshold(arr, threshold_perc=0.75, set_min_v=None, set_max_v=None, title='contour', color_bar_title='', color_scale='Portland', line_width=0.1, height=400, width=600, num_ticks=5,
                          with_details=False):
    # ['Greys', 'YlGnBu', 'Greens', 'YlOrRd', 'Bluered', 'RdBu',
    #            'Reds', 'Blues', 'Picnic', 'Rainbow', 'Portland', 'Jet',
    #            'Hot', 'Blackbody', 'Earth', 'Electric', 'Viridis', 'Cividis']
    arr = np.copy(arr)
    print('MAX BEFORE', np.max(arr))
    print('thresh {}, min {}, max{}'.format(threshold_perc, set_min_v, set_max_v))
    #In order to set a threshold value, set min and max should be None
    if (threshold_perc is not None) and (set_min_v is None) and (set_max_v is None):
        threshold_val = np.max(arr)*threshold_perc
        arr[arr > threshold_val] = threshold_val

    if set_min_v is not None:
        min_v = set_min_v
    else:
        min_v = np.min(arr)
    if set_max_v is not None:
        max_v = set_max_v
    else:
        max_v = np.max(arr)
    #In case we set minimum v
    arr[arr > max_v] = max_v
    if np.sum(arr > max_v) == 0:
        arr[0,0] = max_v

    print('MAX AFTER', np.max(arr))

    tick_x = list(np.arange(min_v, max_v, (max_v-min_v) / (num_ticks)))


    if -np.log10(tick_x[1]) > 0:
        print('Truncating ', int(-np.log10(tick_x[1]))+1)
        print([v for v in tick_x])
        tick_x = [general_utils.truncate(v, int(-np.log10(tick_x[1]))+1) for v in tick_x]
    else:
        print('Truncating', int(-np.log10(tick_x[1])))
        tick_x = [general_utils.truncate(v, int(-np.log10(tick_x[1]))) for v in tick_x]

    if tick_x[1] > 10:
        tick_x = [int(v) for v in tick_x]

    print('TICK X final', tick_x)

    contour = go.Contour(z=arr, x=np.arange(arr.shape[0]), y=np.arange(arr.shape[1]),
                 colorscale=color_scale,
                colorbar=dict(
                    title=color_bar_title,
                    titleside='right',
                    titlefont=dict(
                        size=14,
                        family='Arial, sans-serif',
                    ),
                    tickmode='array',
                    tickvals=tick_x,
                    ticktext=tick_x
                ),
                line=dict(width=line_width, smoothing=0),
            )

    layout = go.Layout(title=title, height=height, width=width)
    fig = go.Figure(data=[contour], layout=layout)

    if with_details:
        return fig, {'min': min_v, 'max' : max_v, 'threshold_perc' : threshold_perc, 'arr' : arr}
    return fig

def plot_contour_multiple(arr_l, title='contour', subplot_titles=None, scale_equal=True, color_bar_title='', color_scale='Portland', line_width=0.1, height=400, width=600, font_size_individual=14):
    contour_l = []
    if subplot_titles is None:
        subplot_titles = [''] * len(arr_l)
    horizontal_spacing = 0.15
    mult = (1 - horizontal_spacing*(len(arr_l)-1)) / len(arr_l)

    if scale_equal:
        max_val = np.max([arr for arr in arr_l])
        min_val = np.min([arr for arr in arr_l])

    for i, arr_i in enumerate(arr_l):
        if scale_equal:
            contour_i = go.Contour(z=arr_i, x=np.arange(arr_i.shape[0]), y=np.arange(arr_i.shape[1]),
                        colorscale=color_scale,
                        colorbar=dict(
                            title=color_bar_title,
                            titleside='right',
                            x=mult*(i+1) + i*horizontal_spacing
                        ),
                        zmin=min_val,
                        zmax=max_val,
                        line=dict(width=line_width, smoothing=0),
                    )
        else:
            contour_i = go.Contour(z=arr_i, x=np.arange(arr_i.shape[0]), y=np.arange(arr_i.shape[1]),
                        colorscale=color_scale,
                        colorbar=dict(
                            title=color_bar_title,
                            titleside='right',
                            x=mult*(i+1) + i*horizontal_spacing
                        ),
                        line=dict(width=line_width, smoothing=0),
                    )
        contour_l.append(contour_i)

    fig = tools.make_subplots(rows=1, cols=len(contour_l), shared_yaxes=False,
                                                            shared_xaxes=False, horizontal_spacing=horizontal_spacing,
                                                            subplot_titles=subplot_titles)
    for i in fig['layout']['annotations']:
        i['font'] = dict(size=font_size_individual)

    for i, contour_i in enumerate(contour_l):
        fig.append_trace(contour_i, 1, i+1)
    fig['layout'].update(height=height,
                        width=width,
                        title=title)
    return fig

def plot_heatmap_contour(arr, title='heatmap-contour', color_bar_title='', color_scale='Portland', height=400, width=600):
    heatmap = go.Heatmap(z=arr, x=np.arange(arr.shape[0]), y=np.arange(arr.shape[1]),
                colorscale=color_scale,
                colorbar=dict(
                    title=color_bar_title,
                    titleside='right',
                    titlefont=dict(
                        size=14,
                        family='Arial, sans-serif',
                    ),
                )
            )
    contour = go.Contour(z=arr, x=np.arange(arr.shape[0]), y=np.arange(arr.shape[1]), showscale= False)

    fig = tools.make_subplots(rows=1, cols=2)
    fig.append_trace(heatmap, 1, 1)
    fig.append_trace(contour, 1, 2)
    fig['layout'].update(height=height,
                        width=width,
                        title=title)
    return fig

def plot_event_triplet(num_events_bins, distances_bins, sizes_bins_lists, durations_bins_lists, num_bins=10, fr=10.3, spatial_res=0.1835, height=400, width=600, title='event-triplet-plot'):
    #Distance from center = x axis
    #Number of events  = y axis
    #Size of event = size of sphere
    #Duration = clock in sphere

    sizes_mean_bins = [np.mean(l) for l in sizes_bins_lists]
    sizes_std_bins = [np.std(l) for l in sizes_bins_lists]
    durations_mean_bins = [np.mean(l/fr) for l in durations_bins_lists]
    durations_std_bins = [np.std(l/fr) for l in durations_bins_lists]

    #print('sizes:',sizes_mean_bins)
    #print('sizes_std:', sizes_std_bins)
    #print('durations:', durations_mean_bins)
    #print('durations_std:', durations_std_bins)
    #print('num events:', num_events_bins)

    max_circle_size = 100
    min_circle_size = 50
    min_colour = np.min(durations_mean_bins)
    max_colour = np.max(durations_mean_bins)
    max_line_width = 7

    trace0 = go.Scatter(
        x=np.array(distances_bins)*spatial_res,
        y=num_events_bins,
        mode='markers',
        marker=dict(
            size = remap_range(old_value_arr=np.array(sizes_mean_bins),
                               old_min=np.min(sizes_mean_bins),
                               old_max=np.max(sizes_mean_bins),
                               new_min=min_circle_size,
                               new_max=max_circle_size),
        cmax=max_colour,
        color= durations_mean_bins,
        colorbar=dict(
            title='Duration (s)'
        ),
        # colorscale  options
        # ['Greys', 'YlGnBu', 'Greens', 'YlOrRd', 'Bluered', 'RdBu',
        #            'Reds', 'Blues', 'Picnic', 'Rainbow', 'Portland', 'Jet',
        #            'Hot', 'Blackbody', 'Earth', 'Electric', 'Viridis', 'Cividis']
        colorscale='Reds',
        line = dict(
          color = 'rgb(205,205, 205)',
          width = (sizes_std_bins / np.max(sizes_std_bins)) * max_line_width
        ),
        opacity=0.8,
        ),
    )

    data = [trace0]
    layout = go.Layout(
                title=title,
                font=dict(family='Courier New, monospace', size=18, color='#494949'),
                yaxis=dict(
                    title='Events over Area',
                ),
                xaxis=dict(
                    title='Radius (\u03bcm)',
                )
            )
    fig = go.Figure(data=data, layout=layout)
    return fig

def plot_point_box(x, y, title='point box', x_title='', y_title='', height=400, width=600, margin_l=100, margin_b=100,
                    y_range=None):
    traces = []

    for xd, yd in zip(x, y):
            traces.append(go.Box(
                y=yd,
                name=xd,
                boxpoints='all',
                jitter=0.3,
                whiskerwidth=0.2,
                marker=dict(
                    size=7,
                ),
                line=dict(width=2),
            ))

    layout = go.Layout(
        title=title,
        yaxis=dict(
            gridcolor='rgb(255, 255, 255)',
            gridwidth=1,
            zerolinecolor='rgb(255, 255, 255)',
            zerolinewidth=2,
            title=y_title
        ),
        xaxis=dict(
            title=x_title
        ),
        margin=dict(
            l=margin_l,
            r=30,
            b=margin_b,
            t=100,
        ),
        paper_bgcolor='rgb(243, 243, 243)',
        plot_bgcolor='rgb(243, 243, 243)',
        showlegend=False
    )

    if y_range is not None:
        layout.update(yaxis=dict(gridcolor='rgb(255, 255, 255)',
                    gridwidth=1,
                    zerolinecolor='rgb(255, 255, 255)',
                    zerolinewidth=2,
                    title=y_title,
                    range=y_range))

    fig = go.Figure(data=traces, layout=layout)
    return fig

def plot_point_box_revised(x, y, title='point box', x_title='', y_title='',
                    height=400, width=600, margin_l=100, margin_b=100, err_type='conf',
                    y_range=None, point_size_mult=1, interval_size_mult=1, return_details=False,
                    ignore_points=False, showlegend=False, squish_x=1, squish_y=1, lines=False,
                    with_stats=False):
    '''
    If we want to have a plot of boxes with x values ['a', 'b', 'c']
    x will be array of size 3 ['a', 'b', 'c']
    y will be list of lists. index 0 will contain a list of the values of a etc
    '''

    print('x', len(x))
    print('y', len(y), len(y[0]))

    for i, (xd, yd) in enumerate(zip(x, y)):
        print('XD {}'.format(i))
        print(xd)
        print(len(yd))

    traces = []
    colour_l = ['rgb(66,244,143)',
                'rgb(66,167,244)',
                'rgb(244,66,167)',
                'rgb(244,143,66)',
                'rgb(244,66,78)',
                'rgb(244,232,66)',
                'rgb(10,10,10)',
                'rgb(255, 102, 102)',
                'rgb(204, 0, 204)',
                'rgb(51, 153, 255)',
                'rgb(153, 0, 76)',
                'rgb(76, 153, 0)',
                'rgb(0, 76, 153)',
                'rgb(153, 153, 153)',
                'rgb(0, 255, 128)',
                'rgb(204, 153, 255)',
                'rgb(204, 204, 0)',
                'rgb(0, 102, 102)']

    if len(x) > len(colour_l):
        colour_l = ['rgb(66,244,143)' for i in range(len(x))]

    mean_l = []
    low_l = []
    high_l = []
    conf_l = []
    std_l = []

    for xd, yd in zip(x, list(y)):
        mean, low, high = stat_utils.mean_confidence_interval(yd, confidence=0.95)
        conf_l.append(high - mean)
        if err_type == 'conf':
            mean_l.append(mean)
            low_l.append(low)
            high_l.append(high)
        else:
            std = np.std(yd)
            std_l.append(std)
            mean_l.append(mean)
            low_l.append(mean-std)
            high_l.append(mean+std)

    boxpoints='all'
    if ignore_points:
        boxpoints = False

    for i, (xd, yd) in enumerate(zip(x, y)):
        if boxpoints:
            traces.append(go.Box(
                pointpos=0,
                y=np.array(yd),
                x=np.array([xd for i in range(len(yd))]),
                boxpoints='all',
                jitter=0.15,
                whiskerwidth=0.2,
                line = dict(color = 'rgba(0,0,0,0)'),
                fillcolor = 'rgba(0,0,0,0)',
                marker=dict(
                    size=7*point_size_mult,
                    color=colour_l[i]
                ),
                showlegend=showlegend
            ))
        else:
            pass

    shapes = []

    print('THE MEANS LIST HEREEE', mean_l)
    max_y = np.max([np.max(high_l), np.max([np.max(y_s) for y_s in y])])
    min_y = np.min([np.min(low_l), np.min([np.min(y_s) for y_s in y])])

    print(min_y)
    print('LOW LIST', low_l)
    print('HIGH LIST', high_l)


    for i in range(len(mean_l)):
        if isinstance(x[i], str):
            p = i
        else:
            p = x[i]

        if len(y[i]) > 1:
            shapes.append(
                {
                'type':'line',
                'x0' : p-0.02 * interval_size_mult,
                'y0' : low_l[i],
                'x1' : p+0.02 * interval_size_mult,
                'y1': low_l[i],
                'opacity' : 0.7,
                'line':  {
                    'color' : 'black',
                    'width' : 1.5,
                },},
            )
            shapes.append(
                {
                'type':'line',
                'x0' : p-0.02 * interval_size_mult,
                'y0' : high_l[i],
                'x1' : p+0.02 * interval_size_mult,
                'y1': high_l[i],
                'opacity' : 0.7,
                'line':  {
                    'color' : 'black',
                    'width' : 1.5,
                },},
            )
            shapes.append(
                {
                'type':'line',
                'x0' : p,
                'y0' : low_l[i],
                'x1' : p,
                'y1': high_l[i],
                'opacity' : 0.7,
                'line':  {
                    'color' : 'black',
                    'width' : 1.5,
                },}
            ),

            range_x = 0.02 * interval_size_mult
            range_y = (range_x/(len(mean_l))) * (np.nanmax([np.nanmax(y_i) for y_i in y])-np.nanmin([np.nanmin(y_i) for y_i in y]))*2
            range_x *= squish_x
            range_y *= squish_y
            #print('RANGE Y', range_y)
            #print('RANGE X', range_x)
            #print('PATH', ' M {} {} L {} {} L {} {} L {} {} Z'.format(p-range_x, mean_l[i], p, mean_l[i]-range_y, p+range_x, mean_l[i], p, mean_l[i]+range_y))
            shapes.append(
                {
                    'type': 'path',
                    'path': ' M {} {} L {} {} L {} {} L {} {} Z'.format(p-range_x, mean_l[i], p, mean_l[i]-range_y, p+range_x, mean_l[i], p, mean_l[i]+range_y),
                    'line': {
                        'color': 'black',
                        'width': 1.5
                    },
                    'fillcolor': colour_l[i],
                },

            )
    print('B')
    if lines:
        for i in range(len(y[0])):
            trace_i = go.Scatter(
                x=x,
                y=[y_sub[i] for y_sub in list(y)],
                mode='lines',
                showlegend=False,
                opacity=0.5,
                line=dict(
                        color='rgba(55, 128, 191, 0.9)',
                        width=2,
                    )
            )
            traces.append(trace_i)
    print('C')
    layout = go.Layout(
        shapes = shapes,
        title=title,
        yaxis=dict(
            gridcolor='rgb(255, 255, 255)',
            gridwidth=1,
            zerolinecolor='rgb(255, 255, 255)',
            zerolinewidth=2,
            title=y_title
        ),
        xaxis=dict(
            title=x_title
        ),
        margin=dict(
            l=margin_l,
            r=30,
            b=margin_b,
            t=100,
        ),
        paper_bgcolor='rgb(255, 255, 255)',
        plot_bgcolor='rgb(255, 255, 255)',
        showlegend=False,
        boxgap=0,
    )

    if return_details:
        return {'traces' : traces, 'shapes' : shapes}

    if y_range is not None:
        layout.update(yaxis=dict(range=y_range))
    else:
        layout.update(yaxis=dict(range=(min_y*(1.05 if min_y <= 0 else 0.95), max_y*1.05)))
    fig = go.Figure(data=traces, layout=layout)

    print('Returning fig???')

    if with_stats:
        return fig, {'x' : x, 'data' : y, 'conf_95' : conf_l, 'std' : std_l, 'mean' : mean_l, 'names' : x}
    return fig


def plot_multi_point_box(x, y, names, title='', x_title='', y_title='', height=400, width=600, margin_l=100, margin_b=100, box_points='outliers'):
    traces = []

    for trace_i in range(len(y)):
        trace = go.Box(
            y=y[trace_i],
            x=x,
            name=names[trace_i],
            whiskerwidth=0.2,
            marker=dict(
                size=7,
            ),
            boxpoints=box_points,
            line=dict(width=2),
        )

        traces.append(trace)

    layout = go.Layout(
        boxmode='group',
        title=title,
        yaxis=dict(
            gridcolor='rgb(255, 255, 255)',
            gridwidth=1,
            zerolinecolor='rgb(255, 255, 255)',
            zerolinewidth=2,
            title=y_title
        ),
        xaxis=dict(
            title=x_title
        ),
        margin=dict(
            l=margin_l,
            r=30,
            b=margin_b,
            t=100,
        ),
        paper_bgcolor='rgb(243, 243, 243)',
        plot_bgcolor='rgb(243, 243, 243)'
    )
    fig = go.Figure(data=traces, layout=layout)
    return fig

def plot_violin_duo(x1, x2, y1, y2, title='', x_title='', y_title=''):
    print('y1', y1)
    print('y2', y2)
    trace_d = {
        "data": [
            {
                "type": 'violin',
                "x": [x1]*len(y1),
                "y": y1,
                "legendgroup": x1,
                "scalegroup": x1,
                "name": x1,
                "side": 'negative',
                "box": {
                    "visible": True
                },
                "meanline": {
                    "visible": True
                },
                "line": {
                    "color": 'blue'
                }
            },
            {
                "type": 'violin',
                "x": [x2]*len(y2),
                "y": y2,
                "legendgroup": x2,
                "scalegroup": x2,
                "name": x2,
                "side": 'positive',
                "box": {
                    "visible": True
                },
                "meanline": {
                    "visible": True
                },
                "line": {
                    "color": 'green'
                }
            }
        ],
    }

    layout = go.Layout(
        title=title,
        yaxis=dict(
            zeroline=False,
            title=y_title
        ),
        xaxis=dict(
            title=x_title
        ),
        paper_bgcolor='rgb(243, 243, 243)',
        plot_bgcolor='rgb(243, 243, 243)',
        violingap=0,
        violinmode='overlay'
    )

    fig = go.Figure(data=trace_d, layout=layout)
    return fig

def remap_range(old_value_arr, old_min, old_max, new_min, new_max):
    old_range = (old_max - old_min)
    if (old_range == 0):
        new_value = new_min
    else:
        new_range = (new_max - new_min)
        new_value = (((old_value_arr - old_min) * new_range) / old_range) + new_min
    return new_value


def copy_fig(fig):
    return go.Figure(fig)


def bin_avg_arr(arr, step_size, bin_type='mean'):
    arr = np.copy(arr)
    rem = len(arr) % step_size
    if rem != 0:
        arr = arr[:-rem]
    arr = np.mean(arr.reshape([-1, step_size]), axis=1)

    if bin_type == 'add':
        arr *= step_size

    return arr

def apply_fun_axis_fig(fig, fun, axis='x'):
    #print(fig)
    if axis == 'x':
        for i in range(len(fig['data'])):
            fig['data'][i]['x'] = fun(fig['data'][i]['x'])
    elif axis == 'y':
        for i in range(len(fig['data'])):
            fig['data'][i]['y'] = fun(fig['data'][i]['y'])
    else:
        print('Invalid axis value in apply_fun_axis_fig. Pass \'x\' or \'y\'')


def seaborn_joint_grid(df, x_key, y_key, kind='reg', title='', text=''):
    plt.cla()   # Clear axis
    plt.clf()   # Clear figure
    sns.set(style='white', font_scale=1.2)
    g = sns.jointplot(x=x_key, y=y_key, data=df, kind=kind, color="xkcd:muted blue")
    g = g.plot_marginals(sns.distplot, kde=True, bins=12, color="xkcd:bluey grey")
    g.ax_joint.text(1.5, 5.5, text, fontstyle='italic')
    plt.tight_layout()
