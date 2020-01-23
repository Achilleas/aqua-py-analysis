function [res] = aqua_cmd_custom_multi(input_folder_path, save_folder_path, preset_id)

tif_files = dir([input_folder_path,'*.tif']);

initial_vars = {'preset_id', 'input_folder_path', 'save_folder_path', 'tif_files', 'i', 'initial_vars'};

for i = 1:numel(tif_files)
    clearvars('-except',initial_vars{:})

    p0 = tif_files(i).folder;
    f0 = tif_files(i).name;
   
    disp(p0);
    disp(f0);
    
    % Get results matrix
    res = aqua_cmd_custom_single(p0, f0, preset_id);
   
    % Update features with GUI commands
    f = aqua_gui_custom(res, 0, 'off');
    
    % In case feature table not loaded maybe
    ui.detect.getFeatureTable(f);
    % Here can insert more GUI commands such as adding borders, masks etc.
    %-----------
    %----------
    % Update features
    ui.detect.updtFeature([],[],f,0);
    res.fts = getappdata(f, 'fts');
    
    % If landmarks exist, add them
    
    % Save results matrix
    save([save_folder_path, f0, '-res.mat'], 'res', '-v7.3');
    disp(res)
end
end

%% Info
% Given folder 'folder_path', process and save the results 'res' 

%input_folder_path = '...';
%save_folder_path = '...';
%preset_id = 2;
