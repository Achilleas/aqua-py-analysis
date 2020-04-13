import pickle as pickle
import imageio as io
import numpy as np
import plotly.io as pio
import plotly.graph_objs as go
import copy
import os
import csv
from analysis.general_utils import plotly_utils

def save_pickle(obj, filename):
    '''
    Save pickle file
    '''
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_pickle(filename):
    '''
    Load pickled file
    '''
    with open(filename, 'rb') as inp:
        p = pickle.load(inp)
        return p

def read_tif_volume(tif_path):
    '''
    Read .tif volume
    '''
    source_vol = io.volread(tif_path).astype(np.float32)
    return source_vol

def write_tif_volume(volume, tif_path, swap=True):
    '''
    Write volume to .tif
    '''
    if swap:
        volume=np.copy(volume)
        volume=np.swapaxes(volume, 1, 2)
    writer = io.get_writer(tif_path, format='TIFF', mode='v', bigtiff=True)
    writer.append_data(volume.astype(np.float32))
    writer.close()

def write_tif_volume_pieces(volume, frames_per_file, tif_path):
    '''
    Write volume to multiple .tif files
    '''
    print('Writing tif files...')
    num_files = volume.shape[0] // frames_per_file
    for i in range(num_files):
        print(i+1)
        volume_piece = volume[i*frames_per_file:(i+1)*frames_per_file]
        write_tif_volume(volume_piece, tif_path + '_' + (i+1) + '.tif')

    if (volume.shape[0] % frames_per_file != 0):
        volume_piece = volume[num_files*frames_per_file:, :, :]
        write_tif_volume(volume_piece, tif_path + '_' + (num_files+1) + '.tif')
    print('Done!')

def save_plotly_fig(fig, image_path, width=1000, height=1000, font_size=None,
                    save_multiple_formats=True, x_title=None, y_title=None):
    '''
    Save plotly figure to image
    '''
    if font_size is None:
        font_size = np.floor(np.mean([width, height]) / 45)
    fig_copy = copy.deepcopy(go.Figure(fig.to_dict()))
    fig_copy['layout'].update(font=dict(family='Times New Roman', size=font_size, color='#494949'),
                         height=height, width=width)
    if isinstance(x_title, str):
        fig_copy['layout'].update(
            xaxis=dict(
            title=x_title
        ))

    if isinstance(y_title, str):
        fig_copy['layout'].update(
            yaxis=dict(
            title=y_title
        ))

    pio.write_image(fig_copy, image_path + '.svg', format='svg',
                    scale=None, width=width, height=height)
    if save_multiple_formats:
        pio.write_image(fig_copy, image_path + '.png', format='png',
                scale=None, width=width, height=height)

def save_csv_dict(d, path, key_order=None):
    with open(os.path.join(path), mode='w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for k in key_order:
            if k in d.keys():
                r = []
                r.append(k)
                r.extend(d[k])
                writer.writerow(r)


def save_pth_plt_l_log(plt_l, pth_l, axis='x'):
    '''
    Given a list of paths and plots:
        Save the plot and its log equivalent (applied on axis specified)
    '''
    for plt, pth in zip(plt_l, pth_l):
        save_plotly_fig(plt, pth)
        plt_copy = plotly_utils.copy_fig(plt)

        if axis=='x':
            plotly_utils.apply_fun_axis_fig(plt_copy, lambda x : np.log(x), axis='x',)

            if 'range' in plt_copy['layout']['xaxis']:
                range_0 = plt_copy['layout']['xaxis']['range'][0]
                range_1 = plt_copy['layout']['xaxis']['range'][1]
                if range_0 is not None:
                    range_0 = np.log(range_0)
                if range_1 is not None:
                    range_1 = np.log(range_1)

            plt_copy['layout']['xaxis']['range'] = [range_0, range_1]
            save_plotly_fig(plt_copy, pth+'_log', x_title=plt_copy['layout']['xaxis']['title']['text'] + '-log')
        else:
            plotly_utils.apply_fun_axis_fig(plt_copy, lambda x : np.log(x), axis='y',)
            print('PLT COPY??', plt_copy)
            print('IS RANGEI N??', 'range' in plt_copy['layout']['yaxis'])
            print('ISsdasf RANGEI N??', 'rangeadsf' in plt_copy['layout']['yaxis'])
            print(plt_copy['layout']['yaxis']['range'], '???')
            if 'range' in plt_copy['layout']['yaxis']:
                if plt_copy['layout']['yaxis']['range'] is not None:
                    print('LAOYT??', plt_copy['layout'])
                    range_0 = plt_copy['layout']['yaxis']['range'][0]
                    range_1 = plt_copy['layout']['yaxis']['range'][1]
                    if range_0 is not None:
                        range_0 = np.log(range_0)
                    if range_1 is not None:
                        range_1 = np.log(range_1)
                    plt_copy['layout']['yaxis']['range'] = [range_0, range_1]

            if plt_copy['layout']['yaxis']['title']['text'] is None:
                save_plotly_fig(plt_copy, pth+'_ylog')
            else:
                save_plotly_fig(plt_copy, pth+'_ylog', y_title=plt_copy['layout']['yaxis']['title']['text'] + '-log')
