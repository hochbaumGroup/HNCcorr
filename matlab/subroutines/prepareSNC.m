function [ simMatrix, compressSimMatrix, sourceWeights, sinkWeights, nTest ] = prepareSNC( s )

% create sparse matrix
simMatrix = sparse( s.idxI, s.idxJ, s.simVal, s.effHeight * s.effWidth, s.effHeight * s.effWidth );

% determine test indices
indices = find( s.test );
nTest = length( indices );

% compress matrix
compressSimMatrix = simMatrix( indices, indices );

% get weights for (s,i)
select = ones( s.effHeight * s.effWidth, 1 );
select( indices ) = 0;
select( s.labels == 0 ) = 0;
sourceWeights = full( sum( simMatrix( logical( select ), indices ), 1 ) );

% get weights for (i,t)
select = ones( s.effHeight * s.effWidth, 1 );
select( indices ) = 0;
select( s.labels == 1  ) = 0;
sinkWeights = full( sum( simMatrix( indices, logical( select ) ), 2 ) )';