function [res] = aqua_cmd_custom_multi_bound(input_folder_path, save_folder_path, cell_bound_path, centre_landmark_path, preset_id)

tif_files = dir([input_folder_path,'*.tif']);

initial_vars = {'centre_landmark_path', 'cell_bound_path', 'preset_id', 'input_folder_path', 'save_folder_path', 'tif_files', 'i', 'initial_vars'};

for i = 1:numel(tif_files)
    clearvars('-except',initial_vars{:})

    p0 = tif_files(i).folder;
    f0 = tif_files(i).name;

    disp(p0);
    disp(f0);

    % Get results matrix
    res = aqua_cmd_custom_single(p0, f0, preset_id, cell_bound_path);

    % Update features with GUI commands
    f = aqua_gui_custom(res, 0, 'off');
    %Load cell boundary and landmark
    bd = getappdata(f, 'bd');

    %Cell boundary
    loadContent = load(cell_bound_path, 'bd0');
    bd('cell') = loadContent.bd0;
    setappdata(f, 'bd', bd);
    ui.movStep(f, [], [], 1);

    %Cell centre landmark
    loadContent = load(centre_landmark_path, 'bd0');
    bd('landmk') = loadContent.bd0;
    setappdata(f, 'bd', bd);
    ui.movStep(f, [], [], 1);

    % In case feature table not loaded maybe
    ui.detect.getFeatureTable(f);
    % Here can insert more GUI commands such as adding borders, masks etc.
    %-----------
    %----------
    % Update features
    ui.detect.updtFeature([],[],f,0);
    res.fts = getappdata(f, 'fts');

    % Save results matrix
    save([save_folder_path, f0, '-res.mat'], 'res', '-v7.3');
    disp(res)
end
end

%% Info
% Given folder 'folder_path', process and save the results 'res'

%Example paths:

%input_folder_path = 'D:\astro_only\m190129_d190226_cx\day_0\registered_videos\';
%save_folder_path = 'D:\astro_only\m190129_d190226_cx\day_0\data\aqua_bound\aqua_res_outputs\';
%cell_bound_path = 'D:\astro_only\m190129_d190226_cx\day_0\data\aqua_bound\aqua_landmarks\cell_bound.mat';
%centre_landmark_path = 'D:\astro_only\m190129_d190226_cx\day_0\data\aqua_bound\aqua_landmarks\cell_centre.mat';
%preset_id = 2;

%input_folder_path = 'D:\astro_only\m190129_d190226_cx\day_2\registered_videos\';
%save_folder_path = 'D:\astro_only\m190129_d190226_cx\day_2\data\aqua_bound\aqua_res_outputs\';
%cell_bound_path = 'D:\astro_only\m190129_d190226_cx\day_2\data\aqua_bound\aqua_landmarks\cell_bound.mat';
%centre_landmark_path = 'D:\astro_only\m190129_d190226_cx\day_2\data\aqua_bound\aqua_landmarks\cell_centre.mat';
%preset_id = 2;

%input_folder_path = 'D:\astro_only\m181129_d190222_c005\day_0\registered_videos\';
%save_folder_path = 'D:\astro_only\m181129_d190222_c005\day_0\data\aqua_bound\aqua_res_outputs\';
%cell_bound_path = 'D:\astro_only\m181129_d190222_c005\day_0\data\aqua_bound\aqua_landmarks\cell_bound.mat';
%centre_landmark_path = 'D:\astro_only\m181129_d190222_c005\day_0\data\aqua_bound\aqua_landmarks\cell_centre.mat';
%preset_id = 2;

%input_folder_path = 'D:\astro_only\m181129_d190222_c005\day_3\registered_videos\';
%save_folder_path = 'D:\astro_only\m181129_d190222_c005\day_3\data\aqua_bound\aqua_res_outputs\';
%cell_bound_path = 'D:\astro_only\m181129_d190222_c005\day_3\data\aqua_bound\aqua_landmarks\cell_bound.mat';
%centre_landmark_path = 'D:\astro_only\m181129_d190222_c005\day_3\data\aqua_bound\aqua_landmarks\cell_centre.mat';
%preset_id = 2;

%input_folder_path = 'D:\astro_only\m181129_d190111_c001\registered_videos1\';
%save_folder_path = 'D:\astro_only\m181129_d190111_c001\data\aqua_bound\aqua_res_outputs\';
%cell_bound_path = 'D:\astro_only\m181129_d190111_c001\data\aqua_bound\aqua_landmarks\cell_bound.mat';
%centre_landmark_path = 'D:\astro_only\m181129_d190111_c001\data\aqua_bound\aqua_landmarks\cell_centre.mat';
%preset_id = 2;

%input_folder_path = 'D:\astro_only\m181129_d190111_c001\registered_videos2\';
%save_folder_path = 'D:\astro_only\m181129_d190111_c001\data\aqua_bound\aqua_res_outputs\';
%cell_bound_path = 'D:\astro_only\m181129_d190111_c001\data\aqua_bound\aqua_landmarks\cell_bound.mat';
%centre_landmark_path = 'D:\astro_only\m181129_d190111_c001\data\aqua_bound\aqua_landmarks\cell_centre.mat';
%preset_id = 2;

%input_folder_path = 'D:\astro_only\m181129_d190111_c001\registered_videos3\';
%save_folder_path = 'D:\astro_only\m181129_d190111_c001\data\aqua_bound\aqua_res_outputs\';
%cell_bound_path = 'D:\astro_only\m181129_d190111_c001\data\aqua_bound\aqua_landmarks\cell_bound.mat';
%centre_landmark_path = 'D:\astro_only\m181129_d190111_c001\data\aqua_bound\aqua_landmarks\cell_centre.mat';
%preset_id = 2;

%input_folder_path = 'F:\190220\190220\astro_only\m2000219_d190411_c003\day_0\registered_videos\';
%save_folder_path = 'F:\190220\190220\astro_only\m2000219_d190411_c003\day_0\data\aqua\aqua_res_outputs\';
%cell_bound_path = 'F:\190220\190220\astro_only\m2000219_d190411_c003\day_0\data\aqua\aqua_landmarks\cell_bound.mat';
%centre_landmark_path = 'F:\190220\190220\astro_only\m2000219_d190411_c003\day_0\data\aqua\aqua_landmarks\cell_centre.mat';
%preset_id = 2;

%input_folder_path = 'F:\190220\190220\astro_only\m2000219_d190411_c003\day_1\registered_videos\';
%save_folder_path = 'F:\190220\190220\astro_only\m2000219_d190411_c003\day_1\data\aqua\aqua_res_outputs\';
%cell_bound_path = 'F:\190220\190220\astro_only\m2000219_d190411_c003\day_1\data\aqua\aqua_landmarks\cell_bound.mat';
%centre_landmark_path = 'F:\190220\190220\astro_only\m2000219_d190411_c003\day_1\data\aqua\aqua_landmarks\cell_centre.mat';
%preset_id = 2;

%input_folder_path = 'F:\190220\190220\astro_only\m2000219_d190411_c004\day_0\registered_videos\';
%save_folder_path = 'F:\190220\190220\astro_only\m2000219_d190411_c004\day_0\data\aqua\aqua_res_outputs\';
%cell_bound_path = 'F:\190220\190220\astro_only\m2000219_d190411_c004\day_0\data\aqua\aqua_landmarks\cell_bound.mat';
%centre_landmark_path = 'F:\190220\190220\astro_only\m2000219_d190411_c004\day_0\data\aqua\aqua_landmarks\cell_centre.mat';
%preset_id = 2;

%input_folder_path = 'F:\190220\190220\astro_only\m2000219_d190411_c004\day_1\registered_videos\';
%save_folder_path = 'F:\190220\190220\astro_only\m2000219_d190411_c004\day_1\data\aqua\aqua_res_outputs\';
%cell_bound_path = 'F:\190220\190220\astro_only\m2000219_d190411_c004\day_1\data\aqua\aqua_landmarks\cell_bound.mat';
%centre_landmark_path = 'F:\190220\190220\astro_only\m2000219_d190411_c004\day_1\data\aqua\aqua_landmarks\cell_centre.mat';
%preset_id = 2;

%%% AXON ONLY IN AXON ASTRO

%input_folder_path = 'E:\astro_axon_green_red\180912_003\registered_videos_astro\';
%save_folder_path = 'E:\astro_axon_green_red\180912_003\data\aqua_bound\aqua_res_outputs\';
%cell_bound_path = 'E:\astro_axon_green_red\180912_003\data\aqua_bound\aqua_landmarks\axon_bound.mat';
%centre_landmark_path = 'E:\astro_axon_green_red\180912_003\data\aqua_bound\aqua_landmarks\landmark_random.mat';
%preset_id = 3;