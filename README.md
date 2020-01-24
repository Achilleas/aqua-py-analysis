Requirements:
Python 3.7, Matlab 2019a (also run in Matlab 2017)

# aqua-py-analysis

Part 1:

Input:
  1) Registered movie in folder with one or more .tif files
    ex.
      - Proprietary video format files (MDF, Sutter Instruments Inc.) were converted to Tiff using commercial software (MView,       Sutter Instrument)
      - Two-photon microscopy movie of astrocyte recording (30.9Hz) 
      - Movement artifact x,y drift correction (translation) using TurgoReg in Fiji (see preprocessing/registration)
      - 1 pixel 3D gaussian filter
      - Moving average z=3 (30.9Hz -> 10.3Hz base framerate)
      - Output folder of registered .tif files (not provided)
      
  2) Run AQuA 
     - **[default - aqua_cmd_custom.m](https://github.com/Achilleas/aqua-py-analysis/blob/master/AQuA-custom/aqua_cmd_custom.m)**
     
     - **[with bounds - cmd_custom_multi_bound.m](https://github.com/Achilleas/aqua-py-analysis/blob/master/AQuA-custom/aqua_cmd_custom_multi_bound.m)**     
     
     Preferably use with cell bound (removes events outside bound) use AqUA interface to generate 2 AqUA landmarks (centre, cell bound)
    If multiple files, merge result files with 
      **[aqua_merge_outputs.ipynb](https://github.com/Achilleas/aqua-py-analysis/blob/master/aqua_merge_outputs.ipynb)**
      
Part 2:
  1) Pre-processed oscilloscope data concatenated in one .csv file (same framerate)
     ex.
      - Rotary encoder analog signal from oscilloscope from MDF
      - Preprocess with: https://github.com/Achilleas/aqua-py-analysis/blob/master/preprocessing/speed.py 
        #python speed.py --input_filepath=$load_path/oscilloscope.txt --output_filepath=$save_path --wheel_radius=7.5 --bin_size=97
  2) Pre-processed behavioural data (pupil, stick, whiskers) in one .csv file (same framerate)
      ex.
       - Behavioural videos from MDF
       - Run https://github.com/Achilleas/aqua-py-analysis/blob/master/preprocessing/get_roi_data_avg_only.ijm
        - Load ROI manager and set ROIs in Fiji (pupil, stick, whiskers)
        - Run script (moving average 10.3Hz and save ROI values) over folder of .tif behaviour videos
        - Merge all .csv files (1 per behaviour video) into a single .csv file 
        https://github.com/Achilleas/aqua-py-analysis/blob/master/preprocessing/merge_roi_folders.py
       
      Result:
      
      | pupil (float)  | stick (bool/float) | whiskers (float) |
      | ------------- | ------------- | -------------|
      | 84.932  | 0  | 2.15 |
      | 84.851  | 0  | 2.158 |
      | ...  | ...  | ... |

Part 3:
  1) Load result file (res.pkl)


**[Processed dataset example](https://drive.google.com/open?id=1AKd6eTaFozHGF5d6zzddp5zLGm503cFs)**
