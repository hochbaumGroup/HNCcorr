function [ finalLabels ] = SNC( s)
n = s.nTest;

completeMat = sparse(n+2,n+2 );
completeMat(1:n,1:n) = s.compressSimMatrix;
completeMat(n+1,1:n) = s.sourceWeights;
completeMat(1:n,n+2) = s.sinkWeights';


arcMatrix = convertToArcMatrix( completeMat, [s.precomputedSum'; 0; 0], zeros(n+2,1), n+1,n+2 );
if s.lambda == -1
    maxLambda = 100000;
    [cuts, s.lambdas ]  = hpf( arcMatrix, n + 2, n + 1, n + 2, [0 maxLambda] , 0 );
    
    nCuts = length( s.lambdas );
    finalLabels = repmat( s.labels, 1, nCuts );
    finalLabels( logical( s.test ),: ) = cuts( 1 : n, : );
else
    [cut, s.lambdas] = hpf( arcMatrix, n + 2, n + 1, n + 2, [s.lambda s.lambda] , 0 );
    
    finalLabels = s.labels;
    finalLabels( logical( s.test ) ) = cut( 1 : n )';
end
