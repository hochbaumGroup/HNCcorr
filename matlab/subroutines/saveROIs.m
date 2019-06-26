function [ ] = saveROIs( s )
    overviewImage = reshape( s.simListMean, s.datasetSize(1), s.datasetSize(2) );
    segmentations = s.segmentations; 
    
    save( [ s.fileNameBase '-ROIs.mat'], 'overviewImage', 'segmentations')
end