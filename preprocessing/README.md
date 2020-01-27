
# Input
-----

## Video preprocessing and AQuA

1. Registered movie in folder with one or more .tif files:
    - example:
      1. Proprietary video format files (MDF, Sutter Instruments Inc.) were converted to Tiff using commercial software (MView,       Sutter Instrument)
      2. Two-photon microscopy movie of astrocyte recording (30.9Hz)
      3. Movement artifact x,y drift correction (translation) using TurgoReg in Fiji (see preprocessing/registration)
      4. 1 pixel 3D gaussian filter
      5. Moving average z=3 (30.9Hz -> 10.3Hz base framerate)
      6. Output folder of registered .tif files (not provided)

2. Run AQuA *(../AQuA-custom)*
     - Edit: **[parameters1.csv](https://github.com/Achilleas/aqua-py-analysis/blob/master/AQuA-custom/cfg/parameters1.csv)** to try a new parameter setting
     - **[default - aqua_cmd_custom_multi.m](https://github.com/Achilleas/aqua-py-analysis/blob/master/AQuA-custom/aqua_cmd_custom_multi.m)**

     - **[with bounds - cmd_custom_multi_bound.m](https://github.com/Achilleas/aqua-py-analysis/blob/master/AQuA-custom/aqua_cmd_custom_multi_bound.m)**     

     Preferably use with cell bound (removes events outside bound) use AQuA interface to generate 2 AqUA landmarks (centre, cell bound)

     #### Parameters:
     ```
     input_folder_path     - folder containing preprocessed .tif files
     save_folder_path      - folder to save results (.mat)
     cell_bound_path       - if available, the astrocyte bound (obtained by running aqua_gui and saving boundary)
     centre_landmark_path  - if available, the centre of the astrocyte (obtained by running aqua_gui and saving landmark at centre)
     preset_id             - From parameters1.csv. 1 for Astrocytes only, 2 for Astro-Axon
     ```

3. Merge resulting files to res.pkl consisting of all event features.
      **[aqua_merge_outputs.ipynb](https://github.com/Achilleas/aqua-py-analysis/blob/master/aqua_merge_outputs.ipynb)**

## Oscilloscope, Pupil, Stick, Whisker preprocessing
  1. Pre-processed oscilloscope data concatenated in one .csv file (same framerate)


  | speed (m/s) |
  | -------------
  | 0.21  |
  | 0.55  |
  | ...  |
     - example
      - Take 8 bit rotary encoder (E6A2, Omron) analog signal from MDF file
  - Run: **[speed.py](https://github.com/Achilleas/aqua-py-analysis/blob/master/preprocessing/speed.py)**
  
        python speed.py --input_filepath=$load_path/oscilloscope.txt --output_filepath=$save_path --wheel_radius=7.5 --bin_size=97
  2. Pre-processed behavioural data (pupil, stick, whiskers) in one .csv file (same framerate)

  | pupil (float)  | stick (bool/float) | whiskers (float) |
  | ------------- | ------------- | -------------|
  | 84.932  | 0  | 2.15 |
  | 84.851  | 0  | 2.158 |
  | ...  | ...  | ... |
      - example
       - Take camera video from MDF
       - Load ROI manager and set ROIs in ImageJ (pupil, stick, whiskers)
       - Run script (moving average 10.3Hz and save ROI values) over folder of .tif behaviour videos to obtain their ROI values.
       Run **[get_roi_data_avg_only.ijm](https://github.com/Achilleas/aqua-py-analysis/blob/master/preprocessing/get_roi_data_avg_only.ijm)** script in ImageJ.

        - Merge all .csv files (1 per behaviour video) into a single .csv file: **[merge_roi_folders.py](
        https://github.com/Achilleas/aqua-py-analysis/blob/master/preprocessing/merge_roi_folders.py)**

## File structure
Look for example structure in datasets provided
