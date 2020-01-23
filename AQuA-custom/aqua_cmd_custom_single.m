function [res] = aqua_cmd_custom_single(p0, f0, preset_id, cell_bound_path)
%% setup
% -- preset 1: in vivo. 2: ex vivo. 3: GluSnFR
startup;  % initialize

%'C:\Users\Leonidas\Desktop\aqua_test_data\aqua_test_folder\'
%p0 = 'C:\Users\Leonidas\Desktop\aqua_test_data\'; % folder name
%f0 = '190111_001_008_reg_8bit_500frame.tif'; % file name

%p0 = '/Users/achilleasgeorgiou/Desktop/aqua_temp/';
%f0 = '190111_001_008_reg_8bit_500frame.tif';
%----------------------------------------------------------------
%----------------------OPTIONS-----------------------------------
%----------------------------------------------------------------
opts = util.parseParam(preset_id, 0);
disp(opts);
%----------------------------------------------------------------
%----------------------------------------------------------------

%% Read data
disp('Reading data');
[datOrg,opts] = burst.prep1(p0,f0,[],opts);  % read data

%% Get border spatial mask
sz = opts.sz;
evtSpatialMask = ones(sz(1),sz(2));
if exist('cell_bound_path', 'var') && ~isempty(cell_bound_path)
    loadContent = load(cell_bound_path, 'bd0');
    bd0 = loadContent.bd0;
    evtSpatialMask = zeros(sz(1),sz(2));
    disp('SpatialMask');
    disp(evtSpatialMask);
    for ii=1:numel(bd0)
        p0 = bd0{ii}{2};
        spaMsk0 = zeros(sz(1),sz(2));
        spaMsk0(p0) = 1;
        evtSpatialMask(spaMsk0>0) = 1;
    end
end

%% Step 1: foreground and seed detection
% detection
disp('Foreground and seed detection');
[dat,dF,arLst,lmLoc,opts,dL] = burst.actTop(datOrg,opts, evtSpatialMask);  % foreground and seed detection

%dat : smoothed datOrg
%dF  : probably df/f?
%dL  : active  voxels?
%arLst : active region list, sets of regions with active voxels?
%lmLoc : local maximums in the movie

%% Step 2: super voxel detection
disp('Super voxel detection');
[svLst,~,riseX] = burst.spTop(dat,dF,lmLoc,evtSpatialMask,opts);  % super voxel detection

%svList : list of cells of super voxels
%riseX  : delay of each super voxel?? (n x 30)
%% Step 3: event detections (also with propagation etc)
disp('Event detection');
[riseLst,datR,evtLst,seLst] = burst.evtTop(dat,dF,svLst,riseX,opts);  % events

% riseLst : rising map list?
% datR    : events visualized in dataset
% evtList : list of events (1 x m cell array o flinear indices
%           corresponding to events)
% seLst   : list of super events (1 x n cell array of linear indices
%           corresponding to super events)
%% Step 4: features?
disp('Features');
[ftsLst,dffMat] = fea.getFeatureQuick(datOrg,evtLst,opts);

% ftsLst : features list with properties
% dffMat : num_features x num_slices (df/f matrix for each of the events
%          found)
%% Step 5: filter by significance level?
% filter by significance level
mskx = ftsLst.curve.dffMaxZ>opts.zThr;
dffMatFilterZ = dffMat(mskx,:);
evtLstFilterZ = evtLst(mskx);
tBeginFilterZ = ftsLst.curve.tBegin(mskx);
riseLstFilterZ = riseLst(mskx);

%% Step 6: merging
% merging (glutamate)
evtLstMerge = burst.mergeEvt(evtLstFilterZ,dffMatFilterZ,tBeginFilterZ,opts);

% reconstruction (glutamate)
if opts.extendSV==0 || opts.ignoreMerge==0 || opts.extendEvtRe>0
    [riseLstE,datRE,evtLstE] = burst.evtTopEx(dat,dF,evtLstMerge,opts);
else
    riseLstE = riseLstFilterZ; datRE = datR; evtLstE = evtLstFilterZ;
end

%% Step 7: feature extraction
% feature extraction
[ftsLstE,dffMatE,dMatE] = fea.getFeaturesTop(datOrg,evtLstE,opts);
ftsLstE = fea.getFeaturesPropTop(dat,datRE,evtLstE,ftsLstE,opts);

% ftsLstE : final features list, (basic, propagation, loc, curve, bds, notes)
% dffMatE : df/f matrix (num_events, num_slices, 2) 1 is with other events, 
%           2 is without any other events 
% dMatE   : value matrix (num_events, num_slices, 2)

%% Step 8: gather results
res = fea.gatherRes(datOrg,opts,evtLstE,ftsLstE,dffMatE,dMatE,riseLstE,datRE);
end