//D:\181022_003\registered
//images_folder = "/Users/achilleasgeorgiou/Dropbox/leo_work/calcium_registration/test_registration/outputs/"
//save_folder_path = "/Users/achilleasgeorgiou/Dropbox/leo_work/calcium_registration/test_registration/roi_data/"

//images_folder = "D:/181012_002/registered/";
//save_folder_path = "D:/181012_002/registered/roi_data/";
//save_folder_path = "C:/Users/Leonidas/Dropbox/leo_work/analysis/data/roi_data_fixed/";

//roi_names_sequentially = newArray("X1", "A1_R2", "A_whole", "allX", "A2", "A3", "A1_R1");

//images_folder= "D:/181022_003/registered";
//save_folder_path = "D:/181022_003/registered/roi_data/";
//roi_names_sequentially = newArray("X1b", "X1", "A1_R1a", "A1_R2", "allX", "X2", "A2_R1", "body", "A1_R1b", "A2_R1b", "X1c");

//images_folder= "D:/180912_003/Registered";
//save_folder_path = "D:/180912_003/Registered/roi_data/";
//roi_names_sequentially = newArray("X1", "A1_a", "A1_b", "A1_c", "A2_far", "A3_far", "X2", "A2", "A1_d", "A2_vessel", "A_whole");


images_folder = "D:/181022_003/registered";
save_folder_path = "D:/181022_003/registered/roi_data2/";
roi_names_sequentially = newArray("X5", "A5_a", "X6", "A6_a", "A6_b", "A5_b", "A7", "A5_c", "A8", "A_whole", "A_body");

//setBatchMode(true);

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
		// 1) Deinterleave red and green channels 
		run("Deinterleave", "how=2 keep");

		for (channel_i=1; channel_i <= 2; channel_i++){
			channel_colour = "";
			if (channel_i == 1) {
				channel_colour = "red";
			}
			else {
				channel_colour = "green";
			}
			selectWindow(filename + " #" + channel_i);
			// 2) Apply gaussian blur 
			run("Gaussian Blur 3D...", "x=1 y=1 z=1");
			// 3) Apply Grouped Z Project
			run("Grouped Z Project...", "projection=[Average Intensity] group=3");
			selectWindow("AVG_" + filename + " #" + channel_i);
			rename("currentImage #" + channel_i);
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
				saveAs("results", save_individual_folder + "/" + roi_names_sequentially[roi_i] + "-" + channel_colour + ".csv"); 
			}
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
