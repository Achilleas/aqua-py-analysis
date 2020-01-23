function [f] = aqua_gui_custom(res,dbg,vis)
    %AQUA_GUI GUI for AQUA

    startup;

    if ~exist('dbg','var')
        dbg = 0;
    end

    f = figure('Name','AQUA','MenuBar','none','Toolbar','none',...
        'NumberTitle','off','Visible','off');

    ui.com.addCon(f,dbg);
    if exist('res','var') && ~isempty(res)
        ui.proj.prep([],[],f,2,res);
    end
    f.Visible = vis;

end
