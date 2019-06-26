function  [ ] = approximatePCA( s )
% Perform an approximate PCA                     
% Author: Quico Spaen & Philipp Baumann
% Date: 14 July 2015
% ------------------------------------------------------------------------
    dataAll = reshape( s.preProcessFunc( ), s.effHeight * s.effWidth, [] );
    
    normRow = sum( dataAll .^2, 2 );
    pRows = normRow ./ sum( normRow ); 
    
    % get frobenius norm of column
    normCol = sum( dataAll .^2 )';
    pCols = normCol ./ sum( normCol );
    
    % capture original rng state
    curRng = rng();
    
    % Seed the random number generator
    rng(s.gridRes);    
    % Determine how many rows and cols to sample
    nRows = ceil( s.effHeight * s.effWidth * s.percRows );
    nRows = max( 150, nRows );
    nCols = ceil( length( normCol ) * s.percCols );
    % sample rows / cols with replacement
    sampleRows = datasample( ( 1 : (s.effHeight * s.effWidth) )', nRows, 1, 'Weights', pRows );
    sampleCols = datasample( ( 1 : length( normCol ) )', nCols, 1, 'Weights', pCols );
    % sort indices
    rowsSelected = sort( sampleRows );
    colsSelected = sort( sampleCols );
    
    % return rng to original stateg
    rng(curRng);

    % obtain sampled data
    % if possible pre assign complete dataset for quick calculation of
    % scores

    dataSample = dataAll( rowsSelected, colsSelected );
    
    % Perform PCA on sample
    [ s.pcaCoeff, ~, latent ] = pca( dataSample );
    s.explainedVar = latent ./ sum( latent );
    
    % Compute scores for original data
    nComps = min( s.nPcaComps, size( s.pcaCoeff, 2 ) );
    % get scores
    pcaScores = dataAll(:, colsSelected) * s.pcaCoeff(:,1: nComps);

    % normalize scores
    maxPca = max( pcaScores);
    minPca = min( pcaScores);
    
    pcaScores = bsxfun( @minus, pcaScores, minPca );
    s.pcaScores = bsxfun( @rdivide, pcaScores, ( maxPca - minPca ) );
end


