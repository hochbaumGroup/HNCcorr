function [ idxIComplete, idxJComplete, simVal, normVal ] = computeNormSim2( s, idxI, idxJ, recomputeFlag , duplicate )
% Compute values of similarities for each row of the vectors (idxI, idxJ) according to the following function 
% simVal(x,y) = exp( - alpha * sqrt( 1/nFrames * norm( x-y)^2 ) ) where
% alpha is a scaling parameter

if recomputeFlag
    y = s.preProcessFunc( );
    y = reshape( y, s.effHeight * s.effWidth, []);
	
    numSim = length( idxI );
    value = zeros( numSim, 1 );
    stepSize = 10000;
    for i = 1 : stepSize : numSim
        value( i : min( i + stepSize - 1, numSim) ) = sum( (y( idxI( i : min( i + stepSize - 1, numSim)),: ) - y( idxJ( i : min( i + stepSize -1, numSim)),: ) ).^ 2, 2) ;
    end
        
	val = value / s.nFramesEff;
	
    % use the fact that similarity values are symmetric to mirror.
    if duplicate==true
        idxIComplete = [ idxI; idxJ ];
        idxJComplete = [ idxJ; idxI ];
        val = [ val; val ];
        [ ~, iSort ] = sort( idxJComplete * ( s.effHeight * s.effWidth - 1 ) + idxIComplete );
        idxIComplete = idxIComplete( iSort );
        idxJComplete = idxJComplete( iSort );
        normVal = val( iSort );    
    else
        idxIComplete = idxI;
        idxJComplete = idxJ;
        normVal = value;
    end
else
    normVal = s.inBetweenVal;
    idxIComplete = idxI;
    idxJComplete = idxJ;
end

simVal = exp( - s.alpha * normVal );
