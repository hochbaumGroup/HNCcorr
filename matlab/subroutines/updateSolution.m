function [ s ] = updateSolution(s)
% Author: Quico Spaen
% ------------------------------------------------------------------------
% Updates current solution to account for new component / tuning parameter
% selection.
% ------------------------------------------------------------------------
% ----- INPUT -----
% d: experimental design. Instance of class ExperimentalDesign
% tuning: Binary variable to indicate whether it is a tuning or testing
%   stage.
% algorithmID .. configID: index for components and tuning configuration.
% s: Instance of class Solution that contains the current of the experiment
%    as well as all data, parameters, and results.
% ----- OUTPUT -----
% s: Updated Solution (see input).
% cpuTotal: Total cpu time used.

% initialize flags

disp( [ 'Segmenting neuron ' num2str( s.roiNumber ) ] )

% find mean intensity over time
tMeanStart = tic;
s.meanMat = getMean( s );
s.cpuGetMean = toc( tMeanStart );

tSimSelectStart = tic;
[ s.idxI, s.idxJ ] = s.simSelectFunc();
s.cpuSimSelect = toc( tSimSelectStart );

tSimFuncStart = tic;
[ s.idxI, s.idxJ, s.simVal, s.inBetweenVal ] = s.simFunc( s.idxI, s.idxJ, true , true);
[ s.simMatrix, s.compressSimMatrix, s.sourceWeights, s.sinkWeights, s.nTest ] = prepareSNC( s );
s.cpuSimFunction = toc( tSimFuncStart );

tQFuncStart = tic;
s.precomputedSum = s.qFunc( true);
s.cpuqFunction = toc( tQFuncStart );

tAlgorithmStart = tic;
s.assignedLabels = s.algFunc();
s.cpuAlgorithm = toc( tAlgorithmStart );
end