from docopt import docopt
import os, glob
import sys
import pandas
import numpy as np

def merge_roi_folders(input_folder, output_file):
    '''
    Assumption: roi folders are sorted alphabetically
    '''
    folder_paths = []
    for folder_path in glob.glob(input_folder + '/*'):
        folder_paths.append(folder_path)
    folder_paths_sorted = sorted(folder_paths)
    file_nums = [int(f_path.split('_')[-1]) for f_path in folder_paths_sorted]
    ind_nums = np.argsort(file_nums)
    folder_paths_sorted = [folder_paths_sorted[ind_nums[i]] for i in range(len(folder_paths_sorted))]

    roi_dict = {'green' : {}, 'red': {}, 'other' : {}}

    for folder_path in folder_paths_sorted:
        for roi_path in glob.glob(folder_path + '/*'):
            print(roi_path)
            filename = os.path.splitext(os.path.basename(roi_path))[0]
            roi_csv = pandas.read_csv(roi_path)
            if ('-green' in filename) or ('-red' in filename):
                roi_name, channel_colour = filename.split('-')
                if roi_name not in roi_dict[channel_colour]:
                    roi_dict[channel_colour][roi_name] = []

                roi_dict[channel_colour][roi_name].extend(roi_csv['Mean'].values)
            else:
                roi_name = filename
                if roi_name not in roi_dict['other']:
                    roi_dict['other'][roi_name] = []
                roi_dict['other'][roi_name].extend(roi_csv['Mean'].values)

    result_green = pandas.DataFrame(data=roi_dict['green'])
    result_red = pandas.DataFrame(data=roi_dict['red'])
    result_other = pandas.DataFrame(data=roi_dict['other'])

    result_green.to_csv(os.path.join(output_file,'green.csv'), index=False)
    result_red.to_csv(os.path.join(output_file, 'red.csv'), index=False)
    result_other.to_csv(os.path.join(output_file, 'other.csv'), index=False)

#python merge_roi_folders.py --input_folder=/Users/achilleasgeorgiou/Dropbox/leo_work/analysis/data/roi_data_fixed --output_filepath=/Users/achilleasgeorgiou/Dropbox/leo_work/analysis/data/roi_merged
if __name__ == '__main__':
    docstr = """Merge roi folders
    Usage:
        merge_roi_folders.py [options]

    Options:
        -h, --help                Print this message
        --base_folder=<str>       Input folder of whole dataset. Script will look under base_folder/data/roi_data for the roi folders
    """

    args = docopt(docstr, version='v0.1')

    base_folder = args['--base_folder']

    input_filepath = os.path.join(base_folder, 'data', 'roi_data')
    output_filepath = os.path.join(base_folder, 'data', 'roi_data_merged')
    print('INPUT:', input_filepath, os.path.exists(input_filepath))
    print('OUTPUT:', output_filepath, os.path.exists(output_filepath))
    merge_roi_folders(input_filepath, output_filepath)
