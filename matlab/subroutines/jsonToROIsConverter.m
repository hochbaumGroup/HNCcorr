function [ s ] = jsonToROIsConverter( jsonFileName, datasetSize, wSize )
    windowSize = wSize;
    halfWindow = floor( windowSize / 2 );
    
    % initialize empty struct
    s = struct();
    
    % load json
    cells = loadjson( jsonFileName );
    characters = str2num(jsonFileName);
    seed = mod(int32(sum(characters)),1024);
    rng(seed);
    permutation = randperm(numel( cells ));
    for j = 1 : numel( cells )
        i = permutation(j);
        s(i).id = i;
        s(i).group = 1;
        
        % determine neuron location
        coordinates = cells{i}.coordinates + 1;
        meanI = floor( mean( coordinates(:,1) ) );
        meanJ = floor( mean( coordinates(:,2) ) );
        
        % determine window shift       
        shiftI = max( 1 - (meanI - halfWindow), 0 ) + min( 0, datasetSize(1) - ( meanI + halfWindow ) );
        shiftJ = max( 1 - (meanJ - halfWindow), 0 ) + min( 0, datasetSize(2) - ( meanJ + halfWindow ) );
        
        minI = meanI - halfWindow + shiftI;
        maxI = meanI + halfWindow + shiftI;
        
        minJ = meanJ - halfWindow + shiftJ;
        maxJ = meanJ + halfWindow + shiftJ;
        
        % mark non neuron pixels as 1 and neuron pixels as 2
        pixelsInWindow = zeros( datasetSize(1), datasetSize(2) );
        pixelsInWindow( minI : maxI, minJ : maxJ ) = 1;
        linInd = sub2ind( datasetSize, coordinates(:,1), coordinates(:,2) );
        pixelsInWindow( linInd ) = 2;
        
        % mark location body        
        s(i).indBody  = find( pixelsInWindow == 2 );
        s(i).indNeuropil  = find( pixelsInWindow == 1 );
    end
    
end
