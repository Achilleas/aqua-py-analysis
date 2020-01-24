function saveMDFvideo2multiTIFF
    warning off
    FileNumber = 0;
    MoreFiles = 1;
    SegmentLength = inputdlg({'Maximal frame number for each TIF file:'},'Decide Stack Length',[1 50],{'400','hsv'});
    SegmentLength = str2num(SegmentLength{1,1});
    while MoreFiles == 1
        FileNumber = FileNumber + 1;
        [file,path] = uigetfile('*.mdf','Select a MDF file');   
        MDFpath{FileNumber} = [path '\' file];
        ID{FileNumber} = file(1:end-4);
        SavePath{FileNumber} = [uigetdir([],'Select Directory to Save') '\'];
        choice = questdlg('Select next file?');
        switch  choice
            case 'Yes'
                MoreFiles = 1;
            case 'No'
                MoreFiles = 0;
            case 'Cancel'
                return
        end
    end
    for N = 1:FileNumber
        f = figure('Name','MDF');
        mfile = actxcontrol('MCSX.Data',[0 0 0 0],f);
        openResult = mfile.invoke('OpenMCSFile',MDFpath{N});
        c = 0;
        while openResult ~= 0
            c = c + 1;
            clear mfile openResult
            close MDF
            mfile = actxcontrol('MCSX.Data',[0 0 0 0],figure('Name','MDF'));
            openResult = mfile.invoke('OpenMCSFile',MDFpath{N});
            if c >= 3
                close MDF
                error('   Please check if the path is incorrect, if the MDF file is already opened elsewhere, or if the MDF file is not exist or dameged.')
            else
            end
        end
        frameCount = str2num(invoke(mfile, 'ReadParameter', 'Video Image Count'));
        if isempty(SegmentLength)
            SegmentLength = frameCount;
        else
        end
        FullSegmentNumber = floor(frameCount/SegmentLength);
        LastFrameCount = mod(frameCount,FullSegmentNumber);
        for segment = 1:FullSegmentNumber
            if LastFrameCount ~= 0
                disp(['ID: ' ID{N} ', file ' num2str(N) ' out of ' num2str(FileNumber) ', segment ' num2str(segment) ' out of ' num2str(FullSegmentNumber+1)])
            else
                disp(['ID: ' ID{N} ', file ' num2str(N) ' out of ' num2str(FileNumber) ', segment ' num2str(segment) ' out of ' num2str(FullSegmentNumber)])
            end
            clear VideoCh
            VideoCh = readMDFvideo( MDFpath{N}, (segment-1)*SegmentLength+1, segment*SegmentLength );
            fprintf('Saving...  ')
            saveMultiTIFF(VideoCh, [SavePath{N} ID{N} '_' num2str(segment) '.tif']);
            if LastFrameCount ~= 0 || segment<FullSegmentNumber
                disp('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
            else
            end
        end
        if LastFrameCount ~= 0
            disp(['ID: ' ID{N} ', file ' num2str(N) ' out of ' num2str(FileNumber) ', segment ' num2str(FullSegmentNumber+1) ' out of ' num2str(FullSegmentNumber+1)])
            clear VideoCh
            VideoCh = readMDFvideo( MDFpath{N}, (FullSegmentNumber)*SegmentLength+1, frameCount );
            fprintf('Saving...  ')
            saveMultiTIFF(VideoCh, [SavePath{N} ID{N} '_' num2str(segment) '.tif']);
        else
        end
    end
    
    function [ VideoCh ] = readMDFvideo( MDFpath, StartFrame, EndFrame )
        %  If StartFrame and EndFrame are not assigned, whole data set will be read.
        warning off
        f = figure('Name','MDF');
        mfile = actxcontrol('MCSX.Data',[0 0 0 0],f);
        openResult = mfile.invoke('OpenMCSFile',MDFpath);
        c = 0;
        while openResult ~= 0
            c = c + 1;
            clear mfile openResult
            close MDF
            mfile = actxcontrol('MCSX.Data',[0 0 0 0],figure('Name','MDF'));
            openResult = mfile.invoke('OpenMCSFile',MDFpath);
            if c >= 3
                close MDF
                error('   Please check if the path is incorrect, if the MDF file is already opened elsewhere, or if the MDF file is not exist or dameged.')
            else
            end
        end
        frameCount = str2num(invoke(mfile, 'ReadParameter', 'Video Image Count'));
        frameHeight = str2num(invoke(mfile, 'ReadParameter', 'Video Height'));
        frameWidth = str2num(invoke(mfile, 'ReadParameter', 'Video Width'));
        fprintf('Creating matrix...  ')
        tic;
        if nargin <= 1
            StartFrame = 1;
            EndFrame = frameCount;
        else
        end
        VideoCh = zeros(frameHeight,frameWidth,EndFrame-StartFrame+1);
        toc;
        fprintf('Read frames...  ')
        tic;
        reverseStr = '';
        for Frame = StartFrame:EndFrame
            p = 100*(Frame-StartFrame+1)/(EndFrame-StartFrame+1);
            msg = sprintf('Percentage done: %3.1f  ', p);
            fprintf([reverseStr, msg]);
            reverseStr = repmat(sprintf('\b'), 1, length(msg));
            V = double(mfile.ReadVideoFrame(Frame)');
            V(V<0) = V(V<0) + 256;
            VideoCh(:,:,(Frame-StartFrame+1)) = abs(V);
        end
        toc;
        close MDF
    end

    function saveMultiTIFF(data, path, options)
        if isa(data, 'double');
            Z = single(data);
            data = uint8(uint8((Z - min(Z(:))) * (255 / (max(Z(:)) - min(Z(:))))));
        else
        end
        tStart = tic;
        errcode = 0;
        try
        if nargin < 3 % Use default options
            options.color = false;
            options.compress = 'no';
            options.message = true;
            options.append = false;
            options.overwrite = false;
        end
        if ~isfield(options, 'message'),   options.message   = true; end
        if ~isfield(options, 'append'),    options.append    = false; end
        if ~isfield(options, 'compress'),  options.compress  = 'no';  end
        if ~isfield(options, 'color'),     options.color     = false; end
        if ~isfield(options, 'overwrite'), options.overwrite = false; end
        if  isfield(options, 'big') == 0,  options.big       = false; end

        if isempty(data), errcode = 1; assert(false); end
        if (options.color == false && ndims(data) > 3) || ...
           (options.color == true && ndims(data) > 4)
            errcode = 2; assert(false);
        end

        if ~options.color
            if ndims(data) >= 4, errcode = 2; assert(false); end;
            [height, width, depth] = size(data);
            tagstruct.Photometric = Tiff.Photometric.MinIsBlack;
        else
            if ndims(data) >= 5, errcode = 2; assert(false); end;
            [height, width, cc, depth] = size(data); % cc: color channels. 3: rgb, 4: rgb with alpha channel
            if cc ~= 3 && cc ~= 4, errcode = 3; assert(false); end;
            tagstruct.Photometric = Tiff.Photometric.RGB;
        end
        tagstruct.ImageLength = height;
        tagstruct.ImageWidth = width;
        tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky; % (RGB RGB,RGB RGB,RGB RGB), http://www.awaresystems.be/imaging/tiff/tifftags/planarconfiguration.html

        if ~options.color && isreal(data) % Grayscale image with real numbers
            tagstruct.SamplesPerPixel = 1;
            data = reshape(data, height, width, 1, depth);
        elseif ~options.color && ~isreal(data) % Grayscale image with complex numbers
            tagstruct.SamplesPerPixel = 2;
            data = reshape([real(data) imag(data)], height, width, 2, depth);
        elseif options.color && isreal(data) % Color image with real numbers
            tagstruct.SamplesPerPixel = cc;
            if cc == 4
                tagstruct.ExtraSamples = Tiff.ExtraSamples.AssociatedAlpha; % The forth channel is alpha channel
            end
            data = reshape(data, height, width, cc, depth);
        elseif options.color && ~isreal(data) % Color image with complex numbers
            tagstruct.SamplesPerPixel = cc * 2;
            if cc == 3
                tagstruct.ExtraSamples = repmat(Tiff.ExtraSamples.Unspecified, 1, 3); % 3(real)+3(imag) = 6 = 3(rgb) + 3(Extra)
            else
                tagstruct.ExtraSamples = repmat(Tiff.ExtraSamples.Unspecified, 1, 5); % 4(real)+4(imag) = 8 = 3(rgb) + 5(Extra)
            end
            data = reshape([real(data) imag(data)], height, width, cc*2, depth);
        end

        switch lower(options.compress)
            case 'no'
                tagstruct.Compression = Tiff.Compression.None;
            case 'lzw'
                tagstruct.Compression = Tiff.Compression.LZW;
            case 'jpeg'
                tagstruct.Compression = Tiff.Compression.JPEG;
            case 'adobe'
                tagstruct.Compression = Tiff.Compression.AdobeDeflate;
            otherwise
                tagstruct.Compression = options.compress;
        end

        switch class(data)
            case {'uint8', 'uint16', 'uint32'}
                tagstruct.SampleFormat = Tiff.SampleFormat.UInt;
            case {'int8', 'int16', 'int32'}
                tagstruct.SampleFormat = Tiff.SampleFormat.Int;
                if options.color
                    errcode = 4; assert(false);
                end
            case {'single', 'double', 'uint64', 'int64'}
                tagstruct.SampleFormat = Tiff.SampleFormat.IEEEFP;
            otherwise
                errcode = 5; assert(false);
        end

        switch class(data)
            case {'uint8', 'int8'}
                tagstruct.BitsPerSample = 8;
            case {'uint16', 'int16'}
                tagstruct.BitsPerSample = 16;
            case {'uint32', 'int32'}
                tagstruct.BitsPerSample = 32;
            case {'single'}
                tagstruct.BitsPerSample = 32;
            case {'double', 'uint64', 'int64'}
                tagstruct.BitsPerSample = 64;
            otherwise
                errcode = 5; assert(false);
        end

        maxstripsize = 8*1024;
        tagstruct.RowsPerStrip = ceil(maxstripsize/(width*(tagstruct.BitsPerSample/8)*size(data,3))); % http://www.awaresystems.be/imaging/tiff/tifftags/rowsperstrip.html
        if tagstruct.Compression == Tiff.Compression.JPEG
            tagstruct.RowsPerStrip = max(16,round(tagstruct.RowsPerStrip/16)*16);
        end

        if exist(path, 'file') && ~options.append
            if ~options.overwrite
                errcode = 6; assert(false);
            end
        end

        path_parent = pwd;
        [pathstr, fname, fext] = fileparts(path);
        if ~isempty(pathstr)
            if ~exist(pathstr, 'dir')
                mkdir(pathstr);
            end
            cd(pathstr);
        end

        file_opening_error_count = 0;
        while ~exist('tfile', 'var')
            try
                if ~options.append % Make a new file
                    s=whos('data');
                    if s.bytes > 2^32-1 || options.big
                        tfile = Tiff([fname, fext], 'w8'); % Big Tiff file
                    else
                        tfile = Tiff([fname, fext], 'w');
                    end
                else
                    if ~exist([fname, fext], 'file') % Make a new file
                        s=whos('data');
                        if s.bytes > 2^32-1 || options.big
                            tfile = Tiff([fname, fext], 'w8'); % Big Tiff file
                        else
                            tfile = Tiff([fname, fext], 'w');
                        end
                    else % Append to an existing file
                        tfile = Tiff([fname, fext], 'r+');
                        while ~tfile.lastDirectory(); % Append a new image to the last directory of an exiting file
                            tfile.nextDirectory();
                        end
                        tfile.writeDirectory();
                    end
                end
            catch
                file_opening_error_count = file_opening_error_count + 1;
                pause(0.1);
                if file_opening_error_count > 5 % automatically retry to open for 5 times.
                    reply = input('Failed to open the file. Do you wish to retry? Y/n: ', 's');
                    if isempty(reply) || any(upper(reply) == 'Y')
                        file_opening_error_count = 0;
                    else
                        errcode = 7;
                        assert(false);
                    end
                end
            end
        end

        for d = 1:depth
            tfile.setTag(tagstruct);
            tfile.write(data(:, :, :, d));
            if d ~= depth
               tfile.writeDirectory();
            end
        end

        tfile.close();
        if exist('path_parent', 'var'), cd(path_parent); end

        tElapsed = toc(tStart);
        if options.message
            display(sprintf('The file was saved successfully. Elapsed time : %.3f s.', tElapsed));
        end

        catch exception
            if exist('tfile', 'var'), tfile.close(); end
            switch errcode
                case 1
                    if options.message, error '''data'' is empty.'; end;
                case 2
                    if options.message, error 'Data dimension is too large.'; end;
                case 3
                    if options.message, error 'Third dimesion (color depth) should be 3 or 4.'; end;
                case 4
                    if options.message, error 'Color image cannot have int8, int16 or int32 format.'; end;
                case 5
                    if options.message, error 'Unsupported Matlab data type. (char, logical, cell, struct, function_handle, class)'; end;
                case 6
                    if options.message, error 'File already exists.'; end;
                case 7
                    if options.message, error(['Failed to open the file ''' path '''.']); end;
                otherwise
                    if exist('fname', 'var') && exist('fext', 'var')
                        delete([fname fext]);
                    end
                    if exist('path_parent', 'var'), cd(path_parent); end
                    rethrow(exception);
            end
            if exist('path_parent', 'var'), cd(path_parent); end
        end
        res = errcode;
    end
    
end

