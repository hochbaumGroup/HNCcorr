function[ y ] = preprocessCorrelation( s, varg )

    if(isempty(varg))
        if(s.coordinates ~= s.coordinatesPreProcess)
            s.coordinatesPreProcess = s.coordinates;
            s.processedData = reshape( corrcoef( reshape( s.dataFile, s.effHeight * s.effWidth,[] )' ), s.effHeight, s.effWidth, [] );
        end
        y = s.processedData;
    else
        coor = varg{1};
        y = h5read( s.dataFileName, '/data', [ coor(1), coor(3), 1 ], [ coor( 2 ) - coor( 1 ) + 1, coor( 4 ) - coor( 3 ) + 1, s.nFramesEff ] );
        y = reshape( corrcoef( reshape( y, ( coor( 2 ) - coor( 1 ) + 1) * ( coor( 4 ) - coor( 3 ) + 1 ),[] )' ), coor( 2 ) - coor( 1 ) + 1, coor( 4 ) - coor( 3 ) + 1, [] );
    end
    k = find(isnan(y));
    y(k) = 0;
end
