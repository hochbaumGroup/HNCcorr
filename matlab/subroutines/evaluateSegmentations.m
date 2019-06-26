function [bestSegmentation, bestSegmentationIndex, rejectionSummary] = evaluateSegmentations( s, assignedLabels )
    % loop over segmentations to update segmentation
    rejectionSummary = zeros(3);
    for i = 1 : size( assignedLabels, 2 )
        cutMat = reshape( assignedLabels(:,i), s.windowSize, s.windowSize );

        % fill in holes in segmentation
        cutMat = fillInHoles( cutMat );
        % find components
        components = matrixComponentSearch( cutMat );

        % find component with most pos seeds.
        seedComp = components( s.posSeeds );
        counts = zeros( max(seedComp), 1  );
        for comp = 1 : max( seedComp )
            counts(comp ) = sum( seedComp==comp );
        end
        [ val, maxCompInd ] = max( counts );
        if val >= floor( length( s.posSeeds ) / 2 ) + 1
            bestComponent = components == maxCompInd;
            [yCell, xCell] = find(bestComponent);
            assignedLabels(:,i) = reshape( bestComponent, [], 1 );

            % check for roundess of shape
            [~,~,singVal] = pca( [ yCell, xCell] );
            %disp(singVal);
            % check if ratio between singular values is not more than 4.            
           if length(singVal)<=1 || (max( singVal(1) / singVal(2), singVal(2) / singVal(1) ) > 10)
               assignedLabels(:,i) = zeros( size( assignedLabels(:,i) ));
               rejectionSummary(1) = rejectionSummary(1)+1;
               continue;
           end
            percPixelsSegmented = sum( assignedLabels(:,i) );
            if percPixelsSegmented < s.minNeuronPixels || percPixelsSegmented > s.maxNeuronPixels
                assignedLabels(:,i) = zeros( size( assignedLabels(:,i) ));
                rejectionSummary(2) = rejectionSummary(2)+1;
                continue;
            end
            height = max(yCell)-min(yCell);
            width = max(xCell)-min(xCell);
            if ( height > 0.95*s.windowSize ) || ( width > 0.95*s.windowSize )
                assignedLabels(:,i) = zeros( size( assignedLabels(:,i) ));
                rejectionSummary(2) = rejectionSummary(2)+1;
                continue;
            end
        else
            assignedLabels(:,i) = zeros( size( assignedLabels(:,i) ));
            rejectionSummary(3) = rejectionSummary(3)+1;
        end
    end

    % determine perc of pixels segmented
    percPixelsSegmented = sum( assignedLabels );
    % pick segmentation with % segmented closest to 15 percent.
    [ ~, bestSegmentationIndex ] = min( (sqrt( s.averageNeuronSize ) - sqrt( percPixelsSegmented  ) ).^2 );
    bestSegmentation = assignedLabels( :, bestSegmentationIndex );
    %disp(reshape (bestSegmentation, s.windowSize,s.windowSize));

end
