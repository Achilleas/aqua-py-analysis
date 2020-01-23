from pystackreg import StackReg
import sys, os, glob
from os.path import basename
import imageio as io
import numpy as np
import multiprocessing
from multiprocessing import Pool, sharedctypes
import time
from itertools import product

def register_folder(source_folder_path, target_file_path, output_folder_path, transformation='translation,center',
                        source_cropping=None, target_cropping=None, parallel=False, num_processes=8):
    """
        Source folder registration consisting of source files given target file path
    """
    t_start = time.time()
    for file_path in glob.glob(os.path.join(source_folder_path, "*.tif")):
        if parallel:
            register_file_multiprocessing(file_path, target_file_path, output_folder_path, transformation=transformation,
                                            source_cropping=None, target_cropping=None, num_processes=num_processes)
        else:
            register_file(file_path, target_file_path, output_folder_path, transformation=transformation, source_cropping=None, target_cropping=None)
    t_end = time.time()
    print("Total time taken: {}s".format(t_end - t_start))

def register_file(source_file_path, target_file_path, output_folder_path, transformation='translation,center',
                        source_cropping=None, target_cropping=None):
    """
        Single file registration
    """
    start_time = time.time()
    source_img = io.volread(source_file_path)
    target_img = io.imread(target_file_path)
    #Initialize registered image
    reg_img = np.zeros(source_img.shape).astype(np.float32)

    t_params = transformation.split(',')
    t_type = t_params[0]
    if t_type == 'translation':
        sr = StackReg(StackReg.TRANSLATION)
    else:
        print('Only translation implemented currently')
        sys.exit()

    #Register each source slice to reference
    num_slices = source_img.shape[0]
    source_img_id = basename(source_file_path).split('.')[0]
    for i in range(num_slices):
        reg_img[i, :, :] = sr.register_transform(target_img[:, :], source_img[i, :, :])
        print("Image: {} Progress: {:6d}/{:6d} \r".format(source_img_id, i+1, num_slices), end='', flush=True),
    print('')
    #Save to folder
    save_path = os.path.join(output_folder_path, source_img_id + '.tif')
    io.volwrite(save_path, reg_img, format='tif')

    print('Time taken (default): {}'.format(time.time() - start_time))

def register_single_multiprocessing(args):
    """
        Single file registration multiprocessing
    """
    sr, i, num_slices, target_img, source_slice = args
    print("Progress: {:6d}/{:6d} \r".format(i+1, num_slices), end='', flush=True),
    return (i, sr.register_transform(target_img, source_slice))

def register_file_multiprocessing(source_file_path, target_file_path, output_folder_path,
        transformation='translation,center', source_cropping=None, target_cropping=None,
        num_processes=8):
    """

    """
    start_time = time.time()

    reader = io.get_reader(source_file_path, format='TIFF', mode='v')
    source_img = reader.get_data(0)
    meta_img = reader.get_meta_data()
    target_img = io.imread(target_file_path)
    #Initialize registered image
    reg_img = np.zeros(source_img.shape).astype(np.float32)

    for k,v in meta_img.items():
        if isinstance(meta_img[k], str):
            meta_img[k] = v.encode('utf-8').strip()

    t_params = transformation.split(',')
    t_type = t_params[0]
    if t_type == 'translation':
        sr = StackReg(StackReg.TRANSLATION)
    else:
        print('Only translation implemented currently')
        sys.exit()

    #Register each source slice to reference
    num_slices = source_img.shape[0]
    source_img_id = basename(source_file_path).split('.')[0]
    print('Image: {}'.format(source_img_id))
    p = Pool(processes=num_processes)
    map_results = p.imap_unordered(register_single_multiprocessing, [(sr, i, num_slices, target_img[:, :], source_img[i, :,: ]) for i in range(num_slices)], chunksize=1)
    p.close()
    p.join()

    #Merge results
    for (i, r) in map_results:
        print("Progress: {:6d}/{:6d} Finishing: {:6d}/{:6d} \r".format(num_slices, num_slices, i+1, num_slices), end='', flush=True),
        reg_img[i, :, :] = r
    print("Progress: {:6d}/{:6d} Finishing: {:6d}/{:6d} \r".format(num_slices, num_slices, num_slices, num_slices), end='', flush=True),
    #Save to folder
    save_path = os.path.join(output_folder_path, source_img_id + '.tif')

    description_str = 'ImageJ=1.52g\nimages={}\nslices={}\nloop=false'.format(num_slices, num_slices)
    meta_img['is_imagej'] = description_str
    if not isinstance(meta_img['compression'], int):
        meta_img['compression'] = meta_img['compression'].value
    meta_img['description'] = description_str
    meta_img['LittleEndian'] = False
    meta_img['is_shaped'] = None
    meta_img['resolution'] = None
    writer = io.get_writer(save_path, format='TIFF', mode='v', bigtiff=True)
    writer.append_data(reg_img)
    writer.close()
    print('Time taken (multiprocessing): {}'.format(time.time() - start_time))
