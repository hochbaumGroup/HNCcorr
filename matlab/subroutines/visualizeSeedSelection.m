function [] = visualizeSeedSelection( s )
    fig1 = figure('visible','off');
    
    imagesc( reshape( s.simListMean, s.datasetSize(1), s.datasetSize(2) ) )
    axis equal tight
    hold on
    for i = 1 : s.numCells
        cell = s.segmentations{ i };
        if cell.assignedLabel
            [ cY,cX ] = find( reshape( cell.bestSegmentation, s.effHeight, s.effWidth ) );
            cX = cX + cell.coordinates(3) - 1;
            cY = cY + cell.coordinates(1) - 1;
            mask = zeros( s.datasetSize(1), s.datasetSize(2) );
            linIndex = sub2ind( s.datasetSize, cY, cX );
            mask( linIndex ) = 1;

            alphamask( mask, [1 0 0], 0.45 );
        end
    end
    
    pos_fig1 = [0 0 1000 1000];
    set(fig1,'Position',pos_fig1)
    set(fig1,'Units','Inches');
    posInch = get(fig1,'Position');
    set(fig1,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[posInch(3), posInch(4)])                                               
                                                
end