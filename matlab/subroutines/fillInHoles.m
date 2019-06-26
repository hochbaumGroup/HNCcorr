function [ segmentation ] = fillInHoles( segmentation )
    matSize = size( segmentation  );
    
    % find components for non-segmented pixels
    componentsInverse= matrixComponentSearch( segmentation == 0 );
    
    boundaryPixels =  [ 1: matSize(1), ...
        1 : matSize(1) : matSize(1) * matSize(2), ...
        matSize(1) : matSize(1) : matSize(1) * matSize(2), ...
        ( matSize(2) - 1 ) * matSize(1) + 1 : matSize(1) * matSize(2) ];
    
    nonHolesComps = unique( componentsInverse( boundaryPixels ) );
    
    for i = 1 : max( componentsInverse(:) )
       if sum( nonHolesComps == i ) == 0
          segmentation( componentsInverse == i ) = 1;
       end
    end
    