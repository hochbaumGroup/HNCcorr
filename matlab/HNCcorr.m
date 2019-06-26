% HNCcorr (main code)
% Authors: Quico Spaen, Roberto Asin
% Date: 2017/04/04

function [ s ] = HNCcorr(conf)

% add to path
addpath( genpath( './subroutines/'));
addpath( genpath( './jsonlab/'));

% compile Mexfiles
compileMexFiles

s = Solution( conf.fileNameBase, conf.dataFileName, conf.dataSize, conf.nFrames, conf.jsonFile, conf.windowSize, conf.minSize, conf.avgSize, conf.maxSize, conf.posSeedsWidth, conf.perc);
s.selectSeeds( );
while s.seedCondition                     
    s.roiNumber = s.roiNumber + 1;
    s.changeWindow( s.roiNumber );
    % update solution
    s = updateSolution( s );
    storeCell(s);
    % select new seeds
    s.selectSeeds( );
end
