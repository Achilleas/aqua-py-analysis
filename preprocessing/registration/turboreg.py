from docopt import docopt
import os, glob
import sys
from register_utils import register_file, register_file_multiprocessing, register_folder

#python calcium_registration/turboreg.py --target=test_registration/ref14.tif --source=test_registration/sources --transformation=translation,center --output=test_registration/python_turboreg_output
#python turboreg.py --source=source_folder_path --target=ref.tif --output=.output_path --parallel --num_processes=40
if __name__ == '__main__':

    docstr = """Turboreg parameters
    Usage:
        turboreg.py [options]

    Options:
        -h, --help                 Print this message
        --source=<str>             Source .tif path. Can be to single .tif file (e.g. folder_path/target.tif) or on folder containing .tif files (e.g. folder_path/)  [default: ]
        --source_cropping=<str>    Source file cropping coordinates (e.g. '40,40,500,500') [default: ]
        --target=<str>             Target (reference) file path .tif.
        --target_cropping=<str>    Target (reference) file cropping coordinates (e.g. '40,40,500,500') [default: ]
        --transformation=<str>     Transformation type with corresponding landmark coordinates. Currently only supporting translation. (but can easily add them if needed)
                                     (e.g. 'translation 50 50 100 100' for source landmark (50, 50), target landmark (100, 100) where (x, y) are coordinates.
                                     Can also write center instead of coordinates to let program automatically choose center as landmark [default: translation,center]
        --output=<str>             Set output folder path [default: turboreg_output]
        --parallel                 If set true, apply registration in parallel w.r.t slices
        --num_processes=<int>      If <parallel> is set, then set the number of processes to run [default: 8]
    """

    args = docopt(docstr, version='v0.1')

    source_path = args['--source']
    target_file_path = args['--target']
    output_folder_path = args['--output']

    source_cropping = args['--source_cropping']
    target_cropping = args['--target_cropping']

    transformation = args['--transformation']
    with_parallel = args['--parallel']
    num_processes = int(args['--num_processes'])

    if os.path.isfile(source_path):
        mode = 'file'
    elif os.path.isdir(source_path):
        mode = 'folder'
    else:
        print('Source does not exist. Invalid file/folder path?')
        sys.exit()

    if not os.path.isfile(target_file_path):
        print('Target file does not exist. Invalid file path?')
        sys.exit()

    #Create folder given folder path
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    print('MODE:', mode)
    if mode == 'file':
        if with_parallel:
            register_file_multiprocessing(source_path, target_file_path, output_folder_path, transformation='translation,center',
                            source_cropping=None, target_cropping=None)
        else:
            register_file(source_path, target_file_path, output_folder_path, transformation='translation,center',
                            source_cropping=None, target_cropping=None)
    elif mode == 'folder':
        register_folder(source_path, target_file_path, output_folder_path, transformation='translation,center',
                        source_cropping=None, target_cropping=None, parallel=with_parallel, num_processes=num_processes)
