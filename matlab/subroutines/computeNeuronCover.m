function [ neuronCover ] = computeNeuronCover( s )
    neuronCover = zeros( s.datasetSize );
    
    for i = 1 : length( s.ROIs )
        if s.ROIs(i).group == 1
            neuronCover( s.ROIs(i).indBody ) = 1;
        end
    end
end