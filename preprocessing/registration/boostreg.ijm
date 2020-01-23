target_filepath = "D:/180912_003/Target/AVG_180912_003_001.tif";
source_folder_path = "D:/180912_003";
save_folder_path = "D:/180912_003/Registered";

//Make windows invinsible (saves processing time and clutter)
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
	//Select final registered stack window, save to specified folder and close
	selectWindow("registeredStack_" + reg_id); saveAs("Tiff", save_folder_path + '/' + basename + '_reg'); close();
}

function get_basename(s){
	dot = indexOf(s, ".");
	if (dot >= 0) s_out = substring(s, 0, dot);
	return s_out;
}
