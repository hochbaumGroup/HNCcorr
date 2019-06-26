function [] = evaluateSeedsAgainstROIs( s )
 

% seedMap index is:
% 0 - no seed
% 1 - seed but not in neuron
% 2 - seed in one neuron
% 3 - seed in multiple neurons
s.seedMap = zeros( s.datasetSize );
s.seedMap( s.potentialSeeds ) = 1;

nSeeds = sum( s.seedMap(:) );
nROIs = length( s.ROIs );
foundROI = zeros( nROIs, 1 );

for i = 1 : length( s.ROIs )
    indicators = s.seedMap( s.ROIs(i).indBody );
    
    % update ROI
    if sum( indicators ) > 0
        foundROI(i) = 1;
    end
    
    % update seed
    s.seedMap( s.ROIs(i).indBody( indicators == 1 ) ) = 2;
    s.seedMap( s.ROIs(i).indBody( indicators == 2 ) ) = 3;
end

if ( min(foundROI) == 0 )
    roiIDNotDetected = extractfield( s.ROIs(logical(1-foundROI)),'id');
else
    roiIDNotDetected = [];
end

f = fopen( [ s.fileNameBase '-seedQuality.txt'], 'w' );

fprintf( f, '# seeds: %d\n', nSeeds );
fprintf( f, 'ROI Recall: %.2f\n', sum( foundROI ) / nROIs * 100 );
fprintf( f, '# Bad seeds: %d\n', sum( s.seedMap(:) == 1 ) );
fprintf( f, '# seeds in multiple neurons: %d\n', sum( s.seedMap(:) == 3 ) );

g=sprintf('%d ', roiIDNotDetected);
fprintf(f, 'ROIs not detected: %s\n', g);

fclose(f);
end