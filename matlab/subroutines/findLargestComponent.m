function [ bestComponent ] = findLargestComponent( components )
sizeMatrix = size( components );
numComponents = max( components(:) );

bestComponent = zeros( sizeMatrix(1), sizeMatrix( 2 ) );
maxComponent = 0;
for index = 1 : numComponents
    count = sum( components(:) == index ) ;
    if count > maxComponent
        bestComponent = components == index;
        maxComponent = count;
    end
end