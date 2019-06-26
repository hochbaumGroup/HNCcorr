function[ y ] = getRawData( s, varg )
    if(isempty(varg))
        y =  h5read( s.dataFileName, '/data', [ s.coordinates(1), s.coordinates(3), 1 ],[ s.coordinates( 2 ) - s.coordinates( 1 ) + 1, s.coordinates( 4 ) - s.coordinates( 3 ) + 1, s.nFramesEff ] );
        y(isnan(y))=0;
    else
        coor = varg{1};
        y = h5read( s.dataFileName, '/data', [ coor(1), coor(3), 1 ],[ coor( 2 ) - coor( 1 ) + 1, coor( 4 ) - coor( 3 ) + 1, s.nFramesEff ] );
        y(isnan(y))=0;
    end
end
