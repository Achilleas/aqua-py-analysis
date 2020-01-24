# aqua-py-analysis

Step 1:

Input:
  1) Movie as .tif file or folder of .tif files
    ex.
      - Proprietary video format files (MDF, Sutter Instruments Inc.) were converted to Tiff using commercial software (MView,       Sutter Instrument)
      - Two-photon microscopy movie of astrocyte recording (30.9Hz) 
      - Movement artifact x,y drift correction (translation) using TurgoReg in Fiji (see preprocessing/registration)
      - 1 pixel 3D gaussian filter
      - Moving average (30.9Hz -> 10.3Hz base framerate)
      - Output folder of registered .tif files (not provided)
  2) Processed oscilloscope, speed data concatenated in one .csv file (10.3Hz)
     ex.
      - https://github.com/Achilleas/aqua-py-analysis/blob/master/preprocessing/speed.py 
        #python speed.py --input_filepath=$load_path/oscilloscope.txt --output_filepath=$save_path --wheel_radius=7.5 --bin_size=97
      - 

| pupil (float)  | stick (bool) | whiskers (float) |
| ------------- | ------------- | -------------|
| 84.932  | 0  | 2.15 |
| 84.851  | 0  | 2.158 |
| ...  | ...  | ... |

rotary encoder analog signals and animal behaviour movies were synchronously recorded using MScan software.


Two-photon microscopy movies (30.9 Hz),  Proprietary video format files (MDF, Sutter Instruments Inc.) were converted to Tiff using commercial software (MView, Sutter Instrument). Time lapse recordings were preprocessed using Fiji. Movement artifact x, y drift was corrected using the TurboReg plugin (using stiff translation) and a custom macro to automate the process. Videos were then visually inspected to confirm movement artifact correction. A 1 pixel 3D gaussian filter was ap- plied to all 2P image stacks (Fiji). A moving average (bin size = 3) was used to average the recorded video (30.9 Hz) down to 10.3 Hz to increase signal to noise ratio (⇡ 1.7x increase).
