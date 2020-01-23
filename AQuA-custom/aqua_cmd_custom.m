%% setup
% -- preset 1: in vivo. 2: ex vivo. 3: GluSnFR
startup;  % initialize

%'C:\Users\Leonidas\Desktop\aqua_test_data\aqua_test_folder\'
p0 = 'C:\Users\Leonidas\Desktop\aqua_test_data\'; % folder name
%f0 = '190111_001_008_reg_8bit.tif'; % file name
f0 = '190111_001_008_reg.tif'; % file name
%p0 = '/Users/achilleasgeorgiou/Desktop/';
%f0 = '190111_001_008_reg_8bit_500frame.tif';
%----------------------------------------------------------------
%----------------------OPTIONS-----------------------------------
%----------------------------------------------------------------
preset = 1;
opts = util.parseParam(preset,0);

%Options
opts.frameRate = 0.097; % seconds per frame
opts.spatialRes = 0.1835; % spatial resolution um per pixel
opts.regMaskGap = 50; % remove border pixels

%Signal
opts.smoXY = 2.0; % smoothing
opts.thrARScl = 4.0; %intensity threshold
opts.minSize = 6;

%Voxel
opts.thrTWScl = 2.0; % temporal cut
opts.thrExtZ = 2.0; % growing z threshold

%Event
opts.cRise = 2; % rising time uncertainty
opts.cDelay = 2; % slowest delay in propagation
opts.gtwSmo = 1.0; % propagation smoothness

%Clean
opts.zThr = 2; % z score threshold events

%Merge
opts.ignoreMerge = 1; % ignore merge
%----------------------------------------------------------------
%----------------------------------------------------------------

%% Read data
disp('Reading data');
[datOrg,opts] = burst.prep1(p0,f0,[],opts);  % read data
%% Step 1: foreground and seed detection
% detection
disp('Foreground and seed detection');
[dat,dF,arLst,lmLoc,opts,dL] = burst.actTop(datOrg,opts);  % foreground and seed detection

%dat : smoothed datOrg
%dF  : probably df/f?
%dL  : active  voxels?
%arLst : active region list, sets of regions with active voxels?
%lmLoc : local maximums in the movie

%% Step 2: super voxel detection
disp('Super voxel detection');
[svLst,~,riseX] = burst.spTop(dat,dF,lmLoc,[],opts);  % super voxel detection

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

%% Step 8: export to GUI
disp('GUI');
res = fea.gatherRes(datOrg,opts,evtLstE,ftsLstE,dffMatE,dMatE,riseLstE,datRE);
aqua_gui(res, 1);

% visualize the results in each step
if 0
    ov1 = plt.regionMapWithData(arLst,datOrg,0.5); zzshow(ov1);
    ov1 = plt.regionMapWithData(svLst,datOrg,0.5); zzshow(ov1);
    ov1 = plt.regionMapWithData(seLst,datOrg,0.5,datR); zzshow(ov1);
    ov1 = plt.regionMapWithData(evtLst,datOrg,0.5,datR); zzshow(ov1);
    ov1 = plt.regionMapWithData(evtLstFilterZ,datOrg,0.5,datR); zzshow(ov1);
    ov1 = plt.regionMapWithData(evtLstMerge,datOrg,0.5,datR); zzshow(ov1);
    [ov1,lblMapS] = plt.regionMapWithData(evtLstE,datOrg,0.5,datRE); zzshow(ov1);
end