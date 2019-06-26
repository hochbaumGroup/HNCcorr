function [ s ] = jsonToSegmentationConverter( jsonFile, backgroundImage )
    
    s = struct();
    s.datasetSize = size( backgroundImage );    
    
    s.simListMean = reshape( backgroundImage, [], 1 );
    
    cells = loadjson( jsonFile );
    
    s.numCells = numel( cells );
    s.segmentations = containers.Map;
    
    for i = 1 : s.numCells
        s.segmentations( num2str(i) ) = cells{i}.coordinates + 1;
    end

end