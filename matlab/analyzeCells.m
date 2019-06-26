function [ ] = analyzeCells( fileName )

load( fileName,'overviewImage','segmentations' );
nSegm = length( segmentations );

%  Create and then hide the GUI as it is being constructed.
f = figure('Visible','off','Position',[100,100,1300,725]);

hslider = uicontrol('Style','slider', ...
    'Position',[600,25,575,25], ...
    'Callback',@slider_callback);

% set the slider range and step size
set(hslider, 'Min', 1);
set(hslider, 'Max', nSegm);
set(hslider, 'Value', 1);
set(hslider, 'SliderStep', [1/(nSegm -1 ), 1/(nSegm-1) ]);

% next and prev Wrong
nextb    = uicontrol('Style','pushbutton',...
             'String','Next Wrong',...
             'Position',[1180,25,70,25],...
            'Callback', @nextb_callback);
prevb    = uicontrol('Style','pushbutton',...
             'String','Prev Wrong',...
             'Position',[525,25,70,25],...
             'Callback', @prevb_callback);

% overview image
ova = axes('Units','Pixels','Position',[112.5,300,300,300]);
ovt = uicontrol('Style','text','String','Overview Image',...
    'Position',[50,610,425,25],'FontSize',12);

% timeseries image
tsa = axes('Units','Pixels','Position',[50,50,425,200]);
tst = uicontrol('Style','text','String','Timeseries Segmented Neuron',...
    'Position',[50,255,425,25],'FontSize',12);

% best segmentation
sega = axes('Units','Pixels','Position',[525,300,300,300]);
segt = uicontrol('Style','text','String','Average Intensity Image',...
    'Position',[525,601,300,25],'FontSize',10);

% preprocessed image
ppa = axes('Units','Pixels','Position',[950,300,300,300]);
ppt = uicontrol('Style','text','String','Preprocessed Average Image',...
    'Position',[950,601,300,25],'FontSize',10);

% proposed segmentations
prop1 = axes('Units','Pixels','Position',[525,100,125,125]);
p1t = uicontrol('Style','text','String','lambda = [0,1]',...
    'Position',[525,55,125,40],'FontSize',10);
prop2 = axes('Units','Pixels','Position',[675,100,125,125]);
p2t = uicontrol('Style','text','String','lambda = [0,1]',...
    'Position',[675,55,125,40],'FontSize',10);
prop3 = axes('Units','Pixels','Position',[825,100,125,125]);
p3t = uicontrol('Style','text','String','lambda = [0,1]',...
    'Position',[825,55,125,40],'FontSize',10);
prop4 = axes('Units','Pixels','Position',[975,100,125,125]);
p4t = uicontrol('Style','text','String','lambda = [0,1]',...
    'Position',[975,55,125,40],'FontSize',10);
prop5 = axes('Units','Pixels','Position',[1125,100,125,125]);
p5t = uicontrol('Style','text','String','lambda = [0,1]',...
    'Position',[1125,55,125,40],'FontSize',10);

% title etc
tt = uicontrol('Style','text','String',sprintf('ROI Overview: %d / %d',1,nSegm),...
    'Position',[325,671,700,50],'FontSize',20);

ct = uicontrol('Style','text','String',sprintf('Coordinates: [%d,%d]',1,2),...
    'Position',[525,645,300,30],'FontSize',12);

pt = uicontrol('Style','text','String','Proposals',...
    'Position',[825,235,125,30],'FontSize',12);

% Change units to normalized so components resize
% automatically.
f.Units = 'normalized';
ova.Units = 'normalized';
sega.Units = 'normalized';
ppa.Units = 'normalized';
prop1.Units = 'normalized';
prop2.Units = 'normalized';
prop3.Units = 'normalized';
prop4.Units = 'normalized';
prop5.Units = 'normalized';
hslider.Units = 'normalized';
ovt.Units = 'normalized';
segt.Units = 'normalized';
ppt.Units = 'normalized';
p1t.Units = 'normalized';
p2t.Units = 'normalized';
p3t.Units = 'normalized';
p4t.Units = 'normalized';
p5t.Units = 'normalized';
tt.Units = 'normalized';
ct.Units = 'normalized';
pt.Units = 'normalized';
tsa.Units = 'normalized';
tst.Units = 'normalized';
prevb.Units = 'normalized';
nextb.Units = 'normalized';

% plot overview image
axes(ova)
imagesc( overviewImage );
hold on
for i = 1 : nSegm
    cell = segmentations{ i };
    if cell.assignedLabel
        [ cY,cX ] = find( reshape( cell.bestSegmentation, size( cell.bestSegmentation, 1 ), size( cell.bestSegmentation, 2 ) ) );
        cX = cX + cell.coordinates(3) - 1;
        cY = cY + cell.coordinates(1) - 1;
        mask = zeros( size(overviewImage) );
        linIndex = sub2ind( size(overviewImage), cY, cX );
        mask( linIndex ) = 1;

        alphamask( mask, [1 0 0], 0.45 );
    end
end
hold off

if nSegm > 0
    % identify wrong segmentations.
    wrongIndicator = zeros( nSegm, 1 );
    wrongNext = zeros( nSegm, 1 );
    wrongPrev = zeros( nSegm, 1 );
    for i = 1 : nSegm
        if segmentations{i}.assignedLabel ~= segmentations{i}.correctLabel
            wrongIndicator(i) = 1;
        end
    end
    currentWrong = 1;
    for i = 1 : nSegm
        wrongPrev( i ) = currentWrong;
        if wrongIndicator(i)
            currentWrong = i;
        end
    end

    currentWrong = nSegm;
    for i = nSegm:-1:1
        wrongNext( i ) = currentWrong;
        if wrongIndicator(i)
            currentWrong = i;
        end
    end

    update_segmentation( segmentations{1} );
end


% Assign the GUI a name to appear in the window title.
f.Name = 'Analyze ROIs';
% Move the GUI to the center of the screen.
movegui(f,'center')
% Make the GUI visible.
f.Visible = 'on';

    function slider_callback(slider,~)
        % Hints: get(hObject,'Value') returns position of slider
        %        get(hObject,'Min') and get(hObject,'Max') to determine...
        slider_value = round( get(slider,'Value') );
        slider.Value = slider_value;
        update_segmentation( segmentations{ slider_value } );
    end

    function prevb_callback(button,~)
            newValue = wrongPrev( floor( hslider.Value ) );
            hslider.Value = newValue;
            update_segmentation( segmentations{ newValue } );
    end

    function nextb_callback(button,~)
            newValue = wrongNext( floor( hslider.Value ) );
            hslider.Value = newValue;
            update_segmentation( segmentations{ newValue } );
    end

    function update_segmentation( currentSegmentation )
        if currentSegmentation.assignedLabel == currentSegmentation.correctLabel
            status = 'CORRECT';
        else
            status = 'WRONG';
        end

        tt.String = sprintf('ROI Overview: %d / %d - Status: %s',currentSegmentation.roiNumber,nSegm, status);
        ct.String = sprintf('Coordinates: [%d,%d]',currentSegmentation.coordinates(1),currentSegmentation.coordinates(3));


        % plot first segmentation
        axes( sega )
        imagesc( currentSegmentation.averageInt );

        mask = currentSegmentation.bestSegmentation;
        if ~isempty( currentSegmentation.correctSegmentation )
            alphamask( currentSegmentation.correctSegmentation, [ 1 0 0], 0.45 );
        end
        alphamask( mask, [0 1 0], 0.45 );

        % plot timeseries
        axes( tsa )
        plot( currentSegmentation.timeseries );
        axis( [ 0, length( currentSegmentation.timeseries ), 0, max( 1, max(currentSegmentation.timeseries )*1.1) ] )

        %plot first preprocessed image
        axes( ppa )
        imagesc( currentSegmentation.averageProcessed );

        % plot proposals:
        nProposals = size( currentSegmentation.proposals,3 );

        axes( prop5 )
        if nProposals >= 5
            imagesc( currentSegmentation.proposals(:,:,5) );
            p5t.String = sprintf( 'lambda = [%.2f,%.2f]', currentSegmentation.lambdas(4),currentSegmentation.lambdas(5) );
        else
            imagesc( zeros(10 ) );
            p5t.String = '';
        end
        axis off

        axes( prop4 )
        if nProposals >= 4
            imagesc( currentSegmentation.proposals(:,:,4) );
            p4t.String = sprintf( 'lambda = [%.2f,%.2f]', currentSegmentation.lambdas(3),currentSegmentation.lambdas(4) );
        else
            imagesc( zeros(10 ) );
            p4t.String = '';
        end
        axis off

        axes( prop3 )
        if nProposals >= 3
            imagesc( currentSegmentation.proposals(:,:,3) );
            p3t.String = sprintf( 'lambda = [%.2f,%.2f]', currentSegmentation.lambdas(2),currentSegmentation.lambdas(3) );
        else
            imagesc( zeros(10 ) );
            p3t.String = '';
        end
        axis off

        axes( prop2 )
        if nProposals >= 2
            imagesc( currentSegmentation.proposals(:,:,2) );
            p2t.String = sprintf( 'lambda = [%.2f,%.2f]', currentSegmentation.lambdas(1),currentSegmentation.lambdas(2) );
        else
            imagesc( zeros(10 ) );
            p2t.String = '';
        end
        axis off

        axes( prop1 )
        imagesc( currentSegmentation.proposals(:,:,1) );
        p1t.String = sprintf( 'lambda = [0,%.2f]', currentSegmentation.lambdas(1) );
        axis off
    end
end
