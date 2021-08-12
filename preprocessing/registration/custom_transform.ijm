//source_folder_path = "F:/190220/190220/astro_only/m2000219_d190411_c004/day_1/videos";
//save_folder_path = "F:/190220/190220/astro_only/m2000219_d190411_c004/day_1/registered_videos";


//Optional transforms to apply before or after registration
pre_transforms = newArray("grouped_z_project");
pre_parameters = newArray("x=1 y=1 z=1", "projection=[Average Intensity] group=3");
post_transforms = newArray()
post_parameters = newArray()

//Make windows invisible (saves processing time and clutter)
setBatchMode(false);

//Create folder if it doesn't exist
if (File.exists(save_folder_path) != true){
		File.makeDirectory(save_folder_path);
}

//All source filepaths in source folder specified
source_filepaths = getFileList(source_folder_path);

// Open target image (only need to open this once as it is the reference image of all source images)
open(target_filepath);

//Register each source file with respect to target reference
for(f_i=0; f_i < lengthOf(source_filepaths); f_i++){
	source_filepath = source_folder_path + '/' + source_filepaths[f_i];
	if(endsWith(source_filepath, ".tif") || endsWith(source_filepath, ".TIF")) {
		// Register individual source target pair
		average_single(source_filepath, save_folder_path);
	}
	break;
}

//Set windows to visible again
setBatchMode(false);
//End










//******************************************************************************
// 							Register single image function
// source_filepath  : the filepath of the source video (.tif)
// save_folder_path : the folder path to save registered source
//******************************************************************************

function average_single(source_filepath, save_folder_path){
	source_name = File.getName(source_filepath);
	basename = get_basename(source_name);
	print("Processing: " + source_name);
	//print("Source filepath:"  + source_filepath);
	//print("Source name:" + source_name);
	//print("Basename: " + basename);

	//Open source window
	open(source_filepath);

	//Get dimensions of source and target
	getDimensions(target_w, target_h, dummy, slice_count, frame_count);
	print("Slice count: " + slice_count + " Frame count: " + frame_count)
	
	run("Make Substack...", "slices="+1+"-"+(slice_count-1));
	rename("substack");
	selectWindow(source_name); close();
	selectWindow("substack");

	// Apply pre-registration transformations
	apply_transformations("substack", pre_transforms, pre_parameters);
    
	//save to specified folder and close
	saveAs("Tiff", save_folder_path + '/' + basename);

    //Close source window
	selectWindow(source_name); close();
    
	close();
}

//******************************************************************************
// 							Register single image function
// window_name      : name of the window to apply transformations
// transform_names  : transformations to apply (supported: gaussian, grouped_z_project)
// parameters       : parameters of each transformation
//******************************************************************************


function apply_transformations(window_name, transform_names, parameters){
	selectWindow(window_name);
	for(i=0; i< lengthOf(transform_names); i++){
		// Apply gaussian blur
		if (transform_names[i] == "gaussian"){
				print("Applying gaussian blur...");
				run("Gaussian Blur 3D...", parameters[i]);
				rename(window_name);
				print("Applied gaussian blur");
		}
		// Apply grouped z project
		else if (transform_names[i] == "grouped_z_project"){
				print("Applying Z Project...");
				run("Grouped Z Project...", parameters[i]);
				selectWindow(window_name); close();
				selectWindow("AVG_" + window_name);
				rename(window_name);
				print("Applied z project");
		}
	}
	selectWindow(window_name);
}

function get_basename(s){
	dot = indexOf(s, ".");
	if (dot >= 0) s_out = substring(s, 0, dot);
	return s_out;
}
