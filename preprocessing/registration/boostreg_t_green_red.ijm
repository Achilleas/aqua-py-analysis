
// 1
target_astro_filepath = "D:/astro_axon_green_red/180912_003/reference_images/REF_ASTRO_AVG_180912_003_007.TIF";
target_axon_filepath = "D:/astro_axon_green_red/180912_003/reference_images/REF_AXON_AVG_180912_003_007.TIF";
//Source folder path containing axons on channel #1, astrocytes on channel #2
source_folder_path = "D:/astro_axon_green_red/180912_003/videos";
save_astro_folder_path = "D:/astro_axon_green_red/180912_003/registered_videos_astro";
save_axon_folder_path = "D:/astro_axon_green_red/180912_003/registered_videos_axon";

/*
// 2
target_astro_filepath = "D:/astro_axon_green_red/181012_002/reference_images/REF_ASTRO_AVG_181012_002_029.TIF";
target_axon_filepath = "D:/astro_axon_green_red/181012_002/reference_images/REF_AXON_AVG_181012_002_029.TIF";
//Source folder path containing axons on channel #1, astrocytes on channel #2
source_folder_path = "D:/astro_axon_green_red/181012_002/videos";
save_astro_folder_path = "D:/astro_axon_green_red/181012_002/registered_videos_astro";
save_axon_folder_path = "D:/astro_axon_green_red/181012_002/registered_videos_axon";
*/

// 3
/*
target_astro_filepath = "D:/astro_axon_green_red/181022_003/reference_images/REF_ASTRO_AVG_181022_003_006.TIF";
target_axon_filepath = "D:/astro_axon_green_red/181022_003/reference_images/REF_AXON_AVG_181022_003_006.TIF";
//Source folder path containing axons on channel #1, astrocytes on channel #2
source_folder_path = "D:/astro_axon_green_red/181022_003/videos";
save_astro_folder_path = "D:/astro_axon_green_red/181022_003/registered_videos_astro";
save_axon_folder_path = "D:/astro_axon_green_red/181022_003/registered_videos_axon";
*/


//Optional transforms to apply before or after registration
pre_transforms = newArray("gaussian", "grouped_z_project");
pre_parameters = newArray("x=1 y=1 z=1", "projection=[Average Intensity] group=3");
post_transforms = newArray();
post_parameters = newArray();

//Make windows invinsible (saves processing time and clutter)
setBatchMode(true);

//Create folder if it doesn't exist (astro)
if (File.exists(save_astro_folder_path) != true){
		File.makeDirectory(save_astro_folder_path);
}

//Create folder if it doesn't exist (axon)
if (File.exists(save_axon_folder_path) != true){
		File.makeDirectory(save_axon_folder_path);
}

//All source filepaths in source folder specified
source_filepaths = getFileList(source_folder_path);

// Open target image (only need to open this once as it is the reference image of all source images)
setBatchMode(false);
open(target_astro_filepath);
open(target_axon_filepath);
setBatchMode(true);

//Register each source file with respect to target reference
for(f_i=0; f_i < lengthOf(source_filepaths); f_i++){
	source_filepath = source_folder_path + '/' + source_filepaths[f_i];
	if(endsWith(source_filepath, ".tif") || endsWith(source_filepath, ".TIF")) {
		//Open source window
		open(source_filepath);
		//Open source video containing both channels
		source_name = File.getName(source_filepath);
		
		//Select source video and deinterleave
		selectWindow(source_name);
		setBatchMode(false);
		run("Deinterleave", "how=2");
		setBatchMode(true);
		target_astro_name = File.getName(target_astro_filepath);
		target_axon_name = File.getName(target_axon_filepath);
		
		source_astro_name = source_name + " #2";
		source_axon_name = source_name + " #1";
		
		// Register channel #2 source target pair
		register_single_window(source_astro_name, target_astro_name, save_astro_folder_path);
		// Register channel #1 source target pair
		register_single_window(source_axon_name, target_axon_name, save_axon_folder_path);
	}
}

//Select and close target image
selectWindow(File.getName(target_astro_filepath)); close();
selectWindow(File.getName(target_axon_filepath)); close();

//Set windows to visible again
setBatchMode(false);
//End










//******************************************************************************
// 							Register single image function
// source_name  : the name of the source video window (.tif)
// target_name  : the name of the target image window (.tif)
// save_folder_path : the folder path to save registered source
//******************************************************************************

function register_single_window(source_name, target_name, save_folder_path){
	basename = get_basename(source_name);
	print("Processing: " + source_name);
	//print("Source filepath:"  + source_filepath);
	//print("Source name:" + source_name);
	print("Basename: " + basename);
	selectWindow(source_name);
	// Apply pre-registration transformations
	apply_transformations(source_name, pre_transforms, pre_parameters);

	//Get dimensions of source and target
	selectWindow(source_name);
	getDimensions(source_w, source_h, dummy, num_slices, dummy);

	selectWindow(target_name);
	getDimensions(target_w, target_h, dummy, dummy, dummy);
	reg_id = "reg";

	// Perform registration for each slice in source.
	for(i=1; i <=num_slices; i++) {
		print("Registering slice: " + i + "/" + num_slices);
		selectWindow(source_name);
		setSlice(i);
	    run("Duplicate...", "title=currentFrame_" + reg_id);

	    run_str = "-align -window currentFrame_" + reg_id + " 0 0 " + (source_w-1) + " " + (source_h-1) + " -window " + target_name + " 0 0 " + (target_w-1) + " " + (target_h-1) + " -translation " + round(source_w/2) + " " + round(source_h/2) + " " + round(target_w/2) + " " + round(target_h/2) + " -showOutput";

		run("TurboReg ", run_str);
		selectWindow("Output");
		setSlice(1);
		if (i == 1){
			run("Duplicate...", "title=registeredStack_" + reg_id);
		}
		else {
	        run("Duplicate...", "title=registeredOutput_" + reg_id);
	        run("Concatenate...", "stack1=registeredStack_" + reg_id + " stack2=registeredOutput_" + reg_id + " title=registeredStack_" + reg_id); 
	    }
        selectWindow("currentFrame_" + reg_id); close();
        selectWindow("Output"); close();
	}
	//Close source window
	selectWindow(source_name); close();
	//Select final registered stack window
	selectWindow("registeredStack_" + reg_id);
	//Apply post registration transforms
	apply_transformations("registeredStack_" + reg_id, post_transforms, post_parameters);
	//save to specified folder and close
	saveAs("Tiff", save_folder_path + '/' + basename + '_reg');
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