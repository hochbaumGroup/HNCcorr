function [] = createJSON( s, filename )
% Creates JSON file "filename" according to Neurofinder format

f = fopen( filename, 'w' );

fprintf( f, '[ ' );
first = true;

for i = 1 : s.numCells
    cell = s.segmentations{ i };
    if cell.assignedLabel
        if first
            fprintf( f, '{ "coordinates": [' );
            first = false;
        else
            fprintf( f, ', { "coordinates": [' );
        end
        
        [ cY, cX ] = find( reshape( cell.bestSegmentation, s.effHeight, s.effWidth ) );
        cX = cX + cell.coordinates(3) - 1;
        cY = cY + cell.coordinates(1) - 1;
        
        % print coordinates of each pixel in cell (0-indexed)
        for j = 1 : length( cX)
            if j < length( cX )
                fprintf( f, '[%d,%d], ',cY(j) - 1, cX(j) - 1 );
            else
                fprintf( f, '[%d,%d]',cY(j) - 1, cX(j) - 1 );
            end
        end

        fprintf( f, '] } ' );
    end
end

fprintf( f, ']' );

fclose( f );

end