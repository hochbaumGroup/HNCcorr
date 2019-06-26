function [ tiffInfo ] = initDataStream(s )
global RAWDATA_DIR

% initialize data structure
nFiles = s.nFiles;
tiffInfo = cell( nFiles, 1 );

baseFileName = [ RAWDATA_DIR '/' s.datasetAbbr '/images/image' ];

for i = 0 : nFiles-1
    fileName = strcat( baseFileName , num2str( i, '%05d' ), '.tiff' );
    tiffInfo{ i+1 } = imfinfo( fileName );
end
end