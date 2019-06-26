function storeCell( s )
    % identify boundary pixels (boundary was artificially added) so they can be
    % excluded for scoring
    [ s.bestSegmentation, s.bestSegmentationIndex , rejectionCounters] = evaluateSegmentations(s, s.assignedLabels);

    s.numCells = s.numCells + 1;

    segm = Segmentation();
    segm.roiNumber = s.roiNumber;

    segm.bestSegmentation = reshape( s.bestSegmentation, s.effHeight, s.effWidth);
    segm.bestLambda = s.lambdas(s.bestSegmentationIndex);

    segm.coordinates = s.coordinates;

    segm.averageInt = mean( s.dataFile, 3 );
    segm.averageProcessed = mean( s.processedData, 3 );

    segm.assignedLabel = max(s.bestSegmentation ) == 1;
    segm.rejectionSummary = sprintf('%d x circular + %d x size + %d x seeds',rejectionCounters(1),rejectionCounters(2),rejectionCounters(3));

    if max( s.bestSegmentation ) == 0
        segm.timeseries = zeros( 1, size( s.dataFile,3) );
    else
        dataVec = reshape( s.dataFile, s.effWidth * s.effHeight, [] );
        segm.timeseries = mean( dataVec( logical(s.bestSegmentation), : ), 1 );
    end
    % find correct label
    firstSeed = min( s.posSeeds );
    [ indI, indJ ] = ind2sub( [s.effHeight s.effWidth], firstSeed );
    centerSeed = sub2ind( s.datasetSize, s.coordinates(1) -1  + indI + s.maskSize, s.coordinates(3) - 1 + indJ + s.maskSize );

    segm.proposals = reshape( s.assignedLabels, s.effHeight, s.effWidth,[] );
    segm.lambdas = s.lambdas;
    if not(strcmp(s.groundTruthJson,''))
        segm.correctLabel = s.neuronCover( centerSeed );
        segm.correctSegmentation = s.neuronCover( s.coordinates(1):s.coordinates(2), s.coordinates(3):s.coordinates(4) );
    else
        segm.correctLabel = 0;
    end

    s.segmentations{ s.numCells } = segm;
end
