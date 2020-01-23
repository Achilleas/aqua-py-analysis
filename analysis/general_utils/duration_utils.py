import numpy as np

def signal_duration_split(all_durations_d, signal_duration_ranges=(1.0/3, 2.0/3, 1)):
    '''
    Split signal durations into 0-33%, 33%-66%, 66%-100% ranges
    '''
    sorted_signal_lengths_arr = np.sort(all_durations_d['default'])
    num_lengths = len(sorted_signal_lengths_arr)
    signal_duration_ranges = (1.0/3, 2.0/3, 1)

    i_short = int(np.floor(signal_duration_ranges[0]*num_lengths))
    i_medium = int(np.floor(signal_duration_ranges[1]*num_lengths))
    i_long = num_lengths

    shortest_signal_lengths = sorted_signal_lengths_arr[0:i_short]
    medium_signal_lengths = sorted_signal_lengths_arr[i_short:i_medium]
    long_signal_lengths = sorted_signal_lengths_arr[i_medium:i_long]

    #x_signal length
    short_sl = shortest_signal_lengths[-1]
    medium_sl = medium_signal_lengths[-1]
    long_sl = long_signal_lengths[-1]

    #x signal range
    short_sr = (0, short_sl)
    medium_sr = (short_sl, medium_sl)
    long_sl = (medium_sl, long_sl)

    #print('short:', short_sr, 'medium:', medium_sr, 'long', long_sl)
    #0 no signal, 1 short signal, 2 medium signal, 3 long signal
    all_durations_class_d = {}

    #Go through all_durations d and change to med, short, long
    for k in all_durations_d.keys():
        if k not in all_durations_class_d:
            all_durations_class_d[k] = np.zeros(all_durations_d[k].shape)

        for i, signal_length in enumerate(all_durations_d[k]):
            if signal_length <= short_sr[1]:
                all_durations_class_d[k][i] = 1
            elif signal_length > short_sr[1] and signal_length <= medium_sr[1]:
                all_durations_class_d[k][i] = 2
            else:
                all_durations_class_d[k][i] = 3
    return all_durations_class_d
