from docopt import docopt
import os, glob
import sys
import pandas
import numpy as np
import pandas
from utils import get_voltage_increase

def prepare_speed_from_file(filepath, wheel_radius, bin_size):
    if skip_interval < 0:
        print('Pass positive integer')
        return None
    oscilloscope = pandas.read_csv(filepath, delimiter='\t').values
    times = oscilloscope[:, 0]
    bphase = oscilloscope[:, 2]
    bphase_bin = (bphase > 0.5).astype(np.int_)
    bphase_bin_increase = get_voltage_increase(bphase_bin)
    max_num_bins = int(len(bphase) // bin_size)
    circumference = 2 * np.pi * wheel_radius
    interval = times[1] - times[0]

    #NOTE: THIS IS INCORRECT. THE CORRECT CALCULATION SHOULD BE
    # bin_speed = sum_bin * 4 * circumference * (0.5/360) to obtain cm/interval_time
    #           (there are 4 voltage states before the next 0->1)
    #
    #bin_speed = sum_bin / (circumference) #speed per bin_size * interval seconds

    #NOTE: USE THIS CORRECT VERSION BUT CHANGE self.speed_values in astroAnalyzer
    bin_speed = sum_bin * circumference * 4 * (0.5/360)
    bin_interval = interval * bin_size

    return bin_interval, bin_speed

#python speed.py --input_filepath=../analysis/data/oscilloscope_fixed_181012_002.txt --output_filepath=../analysis/data/oscilloscope_velocity_fixed_181012_002 --wheel_radius=7.5 --bin_size=97
if __name__ == '__main__':
    docstr = """Speed parameters
    Usage:
        speed.py [options]

    Options:
        -h, --help                   Print this message
        --input_filepath=<str>       File consisting of (time, phase_a, phase_b, framesync)  separated by ('\t')
        --output_filepath=<str>      Filepath of output .csv [default: out]
        --wheel_radius=<float>       Radius of the wheel (cm) [default: 7.5]
        --bin_size=<int>             Bin size [default: 97]

    """
    print('Take a look at the speed calculation in the file. An older incorrect calculation is used, \
            which gets fixed later in the pipeline. Replace with correct calculation in comments in file')
    args = docopt(docstr, version='v0.1')

    input_filepath = args['--input_filepath']
    output_filepath = args['--output_filepath']
    wheel_radius = float(args['--wheel_radius'])
    bin_size = int(args['--bin_size'])

    bin_interval, bin_speed = prepare_speed_from_file(input_filepath, wheel_radius, bin_size)
    speed_df = pandas.DataFrame(data=bin_speed, columns=['speed'])
    print('Time interval of each bin: {}s'.format(bin_interval))
    speed_df.to_csv(output_filepath + '.csv', index=False)
