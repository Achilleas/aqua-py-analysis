///////////////////////////////////////////////////////////////////////////
//roi_names_sequentially = newArray("stick", "pupil", "whiskers");
//base_folder = "D:/astro_only/m181129_d190111_c001";

//roi_names_sequentially = newArray("stick", "pupil", "whiskers");
//base_folder = "D:/astro_only/m181129_d190222_c005/day_0";

//roi_names_sequentially = newArray("stick", "pupil", "whiskers");
//base_folder = "D:/astro_only/m181129_d190222_c005/day_3";

//roi_names_sequentially = newArray("stick", "pupil", "whiskers");
//base_folder = "D:/astro_only/m190129_d190226_cx/day_0";

//roi_names_sequentially = newArray("stick", "pupil", "whiskers");
//base_folder = "D:/astro_only/m190129_d190226_cx/day_2";

//roi_names_sequentially = newArray("stick", "pupil", "whiskers");
//base_folder = "D:/astro_only/m190129_d190226_cx/day_27";

//roi_names_sequentially = newArray("pupil", "whiskers");
//base_folder = "F:/astro_only/m2000219_d190411_c003/day_0";

//roi_names_sequentially = newArray("pupil", "whiskers");
//base_folder = "F:/astro_only/m2000219_d190411_c003/day_1";

//roi_names_sequentially = newArray("pupil", "whiskers");
//base_folder = "F:/astro_only/m2000219_d190411_c004/day_0";

//roi_names_sequentially = newArray("pupil", "whiskers");
//base_folder = "F:/astro_only/m2000219_d190411_c004/day_1";

///////////////////////////////////////////////////////////////////////////

//roi_names_sequentially = newArray("stick", "pupil");
//base_folder = "D:/astro_axon_green_red/181022_003";

//roi_names_sequentially = newArray("stick", "pupil");
//base_folder = "D:/astro_axon_green_red/181012_002";

///////////////////////////////////////////////////////////////////////////
images_folder = base_folder + "/behaviour_videos";
save_folder_path = base_folder + "/data/roi_data";

print(images_folder)
print(save_folder_path)

setBatchMode(true);

//Create folder if it doesn't exist
if (File.isDirectory(save_folder_path) != true){
	File.makeDirectory(save_folder_path);
}

//All source filepaths in source folder specified
source_filepaths = getFileList(images_folder); 

//Get number of rois
num_rois = lengthOf(roi_names_sequentially);

//Get roi data for each file in folder
for(f_i=0; f_i < lengthOf(source_filepaths); f_i++){
	source_filepath = images_folder + '/' + source_filepaths[f_i];
	print(source_filepath);
	if(endsWith(source_filepath, ".tif") || endsWith(source_filepath, ".TIF")) {
		filename = File.getName(source_filepath);
		basename = get_basename(filename);

		//Create directory for each file we see
		save_individual_folder = save_folder_path + '/' + basename + '/';
		if (File.isDirectory(save_individual_folder) != true){
			File.makeDirectory(save_individual_folder);
		}
		
		//print('Filename: ' + filename, 'Basename: ' + basename);
		// Open window
		open(source_filepath);
		selectWindow(filename);
		
		// 3) Apply Grouped Z Project
		run("Grouped Z Project...", "projection=[Average Intensity] group=3");
		selectWindow("AVG_" + filename);
		rename("currentImage");
		// Get dimensions of processed image
		getDimensions(dummy, dummy, dummy, num_slices, dummy);
		
		// 4) Extract ROI data
		for(roi_i=0; roi_i < num_rois; roi_i++){
			roiManager("select", roi_i);
			//clean Results tables from possible previous run
			run("Clear Results");
		    //go through each slice and store measurements on the Results table
		    for (n=1; n<=num_slices; n++) {
		    	setSlice(n);
		    	roiManager("Measure");
		    }
			saveAs("results", save_individual_folder + "/" + roi_names_sequentially[roi_i] + ".csv"); 
		}
		//Close all windows
		close("*");
	}
}

function get_basename(s){
	dot = indexOf(s, "."); 
	if (dot >= 0) s_out = substring(s, 0, dot);
	return s_out;
}
//setBatchMode(false);
