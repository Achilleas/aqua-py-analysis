//target_filepath = "D:/180912_003/Target/AVG_180912_003_001.tif";
//source_folder_path = "D:/180912_003";
//save_folder_path = "D:/180912_003/Registered";

//target_filepath = "H:/181129/REF_MIN_AVG_190111_001_001.tif";
//source_folder_path = "H:/181129/190111_001";
//save_folder_path = "H:/181129/registered";

//target_filepath = "H:/181129/190121_008/REF_AVG_190121_008_001.tif";
//source_folder_path = "H:/181129/190121_008/source";
//save_folder_path = "H:/181129/190121_008/registered";

//target_filepath = "D:/astro_only/181129_190222_005/day_3/reference_images/REF_AVG_190225_005_005.tif";
//source_folder_path = "D:/astro_only/181129_190222_005/day_3/videos";
//save_folder_path = "D:/astro_only/181129_190222_005/day_3/registered_videos";

//target_filepath = "D:/astro_only/m190129_d190226_cx/day_0/reference_images/REF_AVG_190226_008_003.tif";
//source_folder_path = "D:/astro_only/m190129_d190226_cx/day_0/videos";
//save_folder_path = "D:/astro_only/m190129_d190226_cx/day_0/registered_videos";

//target_filepath = "D:/astro_only/m190129_d190226_cx/day_2/reference_images/REF_AVG_190228_006_005.tif";
//source_folder_path = "D:/astro_only/m190129_d190226_cx/day_2/videos";
//save_folder_path = "D:/astro_only/m190129_d190226_cx/day_2/registered_videos";

//target_filepath = "D:/astro_only/m181129_d190111_c001/reference_images/REF_AVG_190111_001_013.tif";
//source_folder_path = "D:/astro_only/m181129_d190111_c001/videos";
//save_folder_path = "D:astro_only/m181129_d190111_c001/registered_videos";

//target_filepath = "D:/astro_only/m190129_d190226_cx/day_27/reference_images/REF_AVG_190325_003_006.tif";
//source_folder_path = "D:/astro_only/m190129_d190226_cx/day_27/videos";
//save_folder_path =   "D:/astro_only/m190129_d190226_cx/day_27/registered_videos";

//target_filepath = "F:/190220/190220/astro_only/m2000219_d190411_c003/day_0/reference_images/REF_AVG_190411_003_010.tif";
//source_folder_path = "F:/190220/190220/astro_only/m2000219_d190411_c003/day_0/videos";
//save_folder_path = "F:/190220/190220/astro_only/m2000219_d190411_c003/day_0/registered_videos";

//target_filepath = "F:/190220/190220/astro_only/m2000219_d190411_c003/day_1/reference_images/REF_AVG_190412_002_009.tif";
//source_folder_path = "F:/190220/190220/astro_only/m2000219_d190411_c003/day_1/videos";
//save_folder_path = "F:/190220/190220/astro_only/m2000219_d190411_c003/day_1/registered_videos";

//target_filepath = "F:/190220/190220/astro_only/m2000219_d190411_c004/day_0/reference_images/REF_AVG_190411_004_009.tif";
//source_folder_path = "F:/190220/190220/astro_only/m2000219_d190411_c004/day_0/videos";
//save_folder_path = "F:/190220/190220/astro_only/m2000219_d190411_c004/day_0/registered_videos";

target_filepath = "F:/190220/190220/astro_only/m2000219_d190411_c004/day_1/reference_images/REF_AVG_190412_001_010.tif";
source_folder_path = "F:/190220/190220/astro_only/m2000219_d190411_c004/day_1/videos";
save_folder_path = "F:/190220/190220/astro_only/m2000219_d190411_c004/day_1/registered_videos";



//Optional transforms to apply before or after registration
pre_transforms = newArray("gaussian", "grouped_z_project");
pre_parameters = newArray("x=1 y=1 z=1", "projection=[Average Intensity] group=3");
post_transforms = newArray()
post_parameters = newArray()

//Make windows invisible (saves processing time and clutter)
setBatchMode(true);

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
		register_single(source_filepath, target_filepath, save_folder_path);
	}
}

//Select and close target image
selectWindow(File.getName(target_filepath)); close();

//Set windows to visible again
setBatchMode(false);
//End










//******************************************************************************
// 							Register single image function
// source_filepath  : the filepath of the source video (.tif)
// target_filepath  : the filepath of the target image (.tif)
// save_folder_path : the folder path to save registered source
//******************************************************************************

function register_single(source_filepath, target_filepath, save_folder_path){
	source_name = File.getName(source_filepath);
	target_name = File.getName(target_filepath);
	basename = get_basename(source_name);
	print("Processing: " + source_name);
	//print("Source filepath:"  + source_filepath);
	//print("Source name:" + source_name);
	//print("Basename: " + basename);

	//Open source window
	open(source_filepath);

	selectWindow(source_name);
	// Apply pre-registration transformations
	apply_transformations(source_name, pre_transforms, pre_parameters);

	//Get dimensions of source and target
	selectWindow(source_name);
	getDimensions(source_w, source_h, dummy, num_slices, dummy);

	selectWindow(target_name);
	getDimensions(target_w, target_h, dummy, dummy, dummy);
	reg_id = "id_" + source_name;

	// Perform registration for each slice in source.
	for(i=1; i <=num_slices; i++) {
		print("Registering slice: " + i + "/" + num_slices);
		selectWindow(source_name);
		setSlice(i);
	    run("Duplicate...", "title=currentFrame_" + reg_id);
	    run_str = "-align -window currentFrame_" + reg_id + " 0 0 " + (source_w-1) + " " + (source_h-1) + " -file " + target_filepath + " 0 0 " + (target_w-1) + " " + (target_h-1) + " -translation " + round(source_w/2) + " " + round(source_h/2) + " " + round(target_w/2) + " " + round(target_h/2) + " -showOutput";

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
