function [ ] = fiveByFiveAllPixels( s, distNeg )

if s.roiNumber == 0
    % record pixels that haven't been segmented
    s.pixelNotSegmented = ones( s.datasetSize(1) * s.datasetSize( 2 ), 1 );
    simListMean = - ones( s.datasetSize( 1 ) * s.datasetSize( 2 ), 1 );
    
    % prep necessary counters
    currentSegment = 0;
    offset = - s.windowSize;
    segment = @(x) ceil( x / s.windowSize );
    maxSegment = segment( s.datasetSize(2) );

    rawData = zeros( s.datasetSize(1) * 2* s.windowSize, s.nFramesEff );

    % assign first block
    [ rawData ] = updateData( rawData, s, currentSegment + 1);

    for i = 1 : ( s.datasetSize(1) * s.datasetSize(2) )
        [ indI, indJ ] = ind2sub( s.datasetSize, i );
        if min( indI, indJ ) > s.maskSize && indI <= s.datasetSize(1) - s.maskSize
            if segment( indJ - s.maskSize ) > currentSegment && currentSegment < ( maxSegment - 1 )
                currentSegment = currentSegment + 1;
                offset = offset + s.windowSize;
                [ rawData ] = updateData( rawData, s, currentSegment + 1);
            end
            [ X, Y] = meshgrid( indI - s.maskSize  : indI + s.maskSize , indJ - s.maskSize - offset : indJ + s.maskSize  - offset);
            neighborsLin = sub2ind( s.datasetSize, X(:), Y(:) );
            correlations = corrcoef(rawData(neighborsLin,:)');
            correlations = correlations( :, floor( length( correlations ) / 2 ) + 1 );
            simListMean( i ) = mean( correlations );
        end
    end
    % assign stats to object
    s.simListMean = simListMean;
    potentialSeeds = zeros(ceil(s.datasetSize(1)*s.datasetSize(1)/25),1);
    numPotentialSeeds =1;
    for i = 3 : 5 : s.datasetSize(1)-s.maskSize-1
        for j = 3 : 5 : s.datasetSize(2)-s.maskSize-1
            [ X, Y] = meshgrid( i - s.maskSize  : i +  s.maskSize, j - s.maskSize : j + s.maskSize );
            neighborsLin = sub2ind( s.datasetSize, X(:), Y(:) );
            [~,I] = max(simListMean(neighborsLin));
            potentialSeeds(numPotentialSeeds) = neighborsLin(I);
            numPotentialSeeds = numPotentialSeeds +1;
        end
    end
    % sort potential seeds by mean activation (descending order).
    potentialSeeds = potentialSeeds(1:numPotentialSeeds-1);
    matrixForSeedSorting = [ simListMean(potentialSeeds), potentialSeeds ];
    [~, sortedIndices ] = sortrows( matrixForSeedSorting, - 1 );
    indexTopPercentil = floor(length(sortedIndices)*s.percentilForSeeds);
    s.potentialSeeds = potentialSeeds( sortedIndices(1:indexTopPercentil) );
    s.potentialSeedIndex = 1;                                    
    evaluateSeedsAgainstROIs( s );
else
%       if s.roiNumber == 200
%           s.seedCondition = false;
%           return
%       end

    % find best cut and mark pixels as segmented
    bestCut = evaluateSegmentations(s, s.assignedLabels);
    if (max(bestCut)==1)
        % fix boundaries
        bestCutMat = reshape( bestCut, s.windowSize, s.windowSize );

        indicesMatrix = nan( s.windowSize + 2 * s.maxShift );
        indicesMatrix( s.maxShift + 1 : end - s.maxShift, s.maxShift + 1 : end - s.maxShift ) = bestCutMat;

        for shiftX = - s.maxShift : s.maxShift
            remainShift = floor( sqrt( s.maxShift^2 - shiftX^2 ) );
            for shiftY = - remainShift: remainShift
                indicesMatrix( s.maxShift + 1 + shiftY : end - s.maxShift + shiftY, s.maxShift + 1 + shiftX : end - s.maxShift + shiftX) = max( bestCutMat,  indicesMatrix( s.maxShift + 1 + shiftY : end - s.maxShift + shiftY, s.maxShift + 1 + shiftX : end - s.maxShift + shiftX) );
            end
        end

        bestCutIndices = find(  indicesMatrix( s.maxShift + 1 : end - s.maxShift, s.maxShift + 1 : end - s.maxShift ) );


        [ indI, indJ ] = ind2sub( [s.windowSize, s.windowSize], bestCutIndices );
        indI = indI -1 + s.coordinates(1);
        indJ = indJ -1 + s.coordinates(3);

        s.pixelNotSegmented( sub2ind( s.datasetSize, indI, indJ ) ) = 0;
    else
        [ indI, indJ ] = ind2sub( [s.windowSize, s.windowSize], s.posSeeds );
        indI = indI -1 + s.coordinates(1);
        indJ = indJ -1 + s.coordinates(3);
        s.pixelNotSegmented( sub2ind( s.datasetSize, indI, indJ ) ) = 0;
    end
end

seedNotSelected = true;
while seedNotSelected && (s.potentialSeedIndex <= length(s.potentialSeeds))
    % check if seed is selected
    if s.pixelNotSegmented( s.potentialSeeds( s.potentialSeedIndex ) ) == 1
        indexSeed = s.potentialSeeds( s.potentialSeedIndex );
        seedNotSelected = false;
        % increment index
        s.potentialSeedIndex = s.potentialSeedIndex + 1;
    else
        % increment index
        s.potentialSeedIndex = s.potentialSeedIndex + 1;
    end
end

%test if no more seeds are possible
if seedNotSelected
    s.seedCondition = false;
    return
end

[ indI, indJ ] = ind2sub( s.datasetSize, indexSeed );

halfWindow = floor( s.windowSize / 2 );
center = halfWindow + 1;

% check if boundaries are exceeded and shift windows if necessary
shiftI = max( 1 - (indI - halfWindow), 0 ) + min( 0, s.datasetSize(1) - ( indI + halfWindow ) );
shiftJ = max( 1 - (indJ - halfWindow), 0 ) + min( 0, s.datasetSize(2) - ( indJ + halfWindow ) );

s.coordinates = [ indI - halfWindow + shiftI, indI + halfWindow + shiftI, indJ - halfWindow + shiftJ, indJ + halfWindow + shiftJ];
    
seedPixelI = ( indI - s.coordinates(1) + 1 );
seedPixelJ = ( indJ - s.coordinates(3) + 1 );
[ X, Y] = meshgrid( max(seedPixelI - s.posSeedsSize, 1)  : min(seedPixelI + s.posSeedsSize, s.windowSize) , max(seedPixelJ - s.posSeedsSize, 1)  : min(seedPixelJ + s.posSeedsSize, s.windowSize) );
s.posSeeds = sub2ind( [ s.windowSize s.windowSize ] , X(:), Y(:) );

s.negSeeds = zeros( s.numNegSeeds, 1 );

if( s.detNegSeeds && seedPixelI>distNeg+s.maskSize && seedPixelI < (s.windowSize-distNeg-s.maskSize) && seedPixelJ>distNeg+s.maskSize && seedPixelJ < (s.windowSize-distNeg-s.maskSize) )
   angleShift = floor(360/s.numNegSeeds);
   currentAngle = 0;
   for i=1:s.numNegSeeds
       iCoord = seedPixelI + floor(sin(currentAngle) * distNeg);
       jCoord = seedPixelJ + floor(cos(currentAngle) * distNeg);
       s.negSeeds(i) = sub2ind([ s.windowSize s.windowSize ],iCoord,jCoord);
       currentAngle = currentAngle + angleShift;
   end
else
    randPermutation = randperm( s.windowSize^2 );
    cont = 1;
    for i=1:length(randPermutation)
        [locI, locJ] = ind2sub( [ s.windowSize s.windowSize ] , randPermutation(i));
        points = [ seedPixelI, seedPixelJ; locI, locJ ];
        d= pdist(points,'euclidean');
        if(d > distNeg)
            s.negSeeds(cont) = randPermutation(i);
            cont = cont+1;
            if(cont == s.numNegSeeds+1)
                break;
            end
        end
    end
end

s.labels = -ones( s.windowSize ^ 2 , 1 );
s.test = ones( s.windowSize ^ 2, 1 );

s.labels( s.posSeeds ) = 1;
s.labels( s.negSeeds ) = 0;
s.test( s.posSeeds ) = 0;
s.test( s.negSeeds ) = 0;

end

function [ rawData ] = updateData( rawData, s, currentSegment )
% reshape to matrix
rawData =  reshape( rawData, s.datasetSize(1), 2 * s.windowSize, s.nFramesEff );
% move old data to first block.
rawData( :, 1 : s.windowSize, : ) = rawData( :, s.windowSize + 1 : end, : );
% load new data
if currentSegment == ceil( s.datasetSize(2) / s.windowSize )
    x = s.getDataForSeedSelection( [1, s.datasetSize(1), (currentSegment - 1) * s.windowSize + 1, s.datasetSize(2) ] );
    rawData( :, s.windowSize + 1 : s.windowSize + size( x, 2 ), : ) = x;
else
    rawData( :, s.windowSize + 1 : end, : ) = s.getDataForSeedSelection( [1, s.datasetSize(1), (currentSegment - 1) * s.windowSize + 1, currentSegment * s.windowSize ] );
end
rawData = reshape( rawData, s.datasetSize(1) * 2 * s.windowSize, s.nFramesEff );
end
