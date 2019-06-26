function [ idxI, idxJ ] = sparseComputation( s )
    % get pcascores
    approximatePCA( s );
    
    % get boxes 
    [ s.boxes, s.neighborBoxes, s.boxMapping ] = getBoxes( double(s.pcaScores), s.gridRes, double(min( s.pcaScores )), double(max( s.pcaScores )) );
    
    numDist = 0;
    % Compute number of similarities
    for i=1:size(s.boxes,1)
        numDist = numDist + size(s.boxes{i},2)*(size(s.boxes{i},2)-1)/2;
    end
    for i=1:size(s.neighborBoxes,1) % times 2 for mirroring
       numDist = numDist + size(s.neighborBoxes{i,1},2)*size(s.neighborBoxes{i,2},2);
    end
    
    
    % get idxI and idxJ
    idxI = zeros(numDist,1);
    idxJ = zeros(numDist,1);
    counter = 1;
    
    % Compute indices for objects in same box
    for i=1:size(s.boxes,1)
        % Compute distances
        n = size(s.boxes{i},2)*(size(s.boxes{i},2)-1)/2;

        % Set indices
        m = ceil(sqrt(2*n));
        I = 1:n;
        CI = m - round( sqrt( 2*(1 + n - I) ) );
        RI = mod(I + CI.*(CI+1)/2 - 1, m) + 1;  
        idxI(counter:counter+n-1) = [ s.boxes{i}(CI) ];
        idxJ(counter:counter+n-1) = [ s.boxes{i}(RI) ];

        counter = counter + n;
    end

    % Compute indices for objects in neighboring s.boxes
    for i=1:size(s.neighborBoxes,1)

        % Compute distances        
        len = size(s.neighborBoxes{i,1},2)*size(s.neighborBoxes{i,2},2);

        % Set indices
        m = size(s.neighborBoxes{i,1},2);
        I = 1:len;
        CI = ceil(I./m);
        RI = mod((I-1),m)+1;  
        idxJ(counter:counter+len-1) = [ s.neighborBoxes{i,2}(CI) ];   
        idxI(counter:counter+len-1) = [ s.neighborBoxes{i,1}(RI) ];

        counter = counter + len;        
    end

end