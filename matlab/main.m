%CONFIGURATION VALUES
%---------------------

% see README.md for instructions
conf = struct();

% setup parameters 
conf.fileNameBase = 'neurofinder.02.00'; 
conf.dataFileName = 'H5files/neurofinder.02.00_10.h5'; 
conf.dataSize = [512,512]; 
conf.nFrames = 800;
conf.jsonFile = '';
conf.windowSize = 31; 

% seed selection parameters
conf.perc = 0.4; 
conf.posSeedsWidth = 0; 

% oracle parameters
conf.minSize = 40; 
conf.avgSize = 80; 
conf.maxSize = 200; 

%---------------------

s = HNCcorr(conf);

% create json file
createJSON( s, [ s.fileNameBase '-segmentations.json' ] )

% compare against reference segmentation
% if not( strcmp( s.groundTruthJson,'') )
%     disp('Evaluate segmentations with NeuroFinder')
%     system( [ 'neurofinder evaluate "' s.groundTruthJson '" "' s.fileNameBase '-segmentations.json"'] );
%     system( [ 'neurofinder evaluate "' s.groundTruthJson '" "' s.fileNameBase '-segmentations.json"' ' > "' s.fileNameBase '-neurofinderResults.txt"' ] );
% end

% save output as MAT-file
saveROIs( s )
