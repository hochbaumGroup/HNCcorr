function [ components ] = matrixComponentSearch( indicatorMatrix )

sizeMatrix = size( indicatorMatrix );
nElements = prod( sizeMatrix );
components = zeros( sizeMatrix( 1 ), sizeMatrix( 2 ) );

currentComponent = 0;

indices = find( indicatorMatrix );
currentIndex = 1;

while currentIndex <= length( indices )
   if indicatorMatrix( indices( currentIndex ) ) == 1
       currentComponent = currentComponent + 1;
       
       % create queue
       queue = zeros( nElements, 1  );
       queueIndex = 1;
       
       % handle first element
       queue( queueIndex ) = indices( currentIndex );
       indicatorMatrix( indices( currentIndex ) ) = 0;
       components( indices( currentIndex ) )  = currentComponent;
       
       while queueIndex > 0
           currentNode = queue( queueIndex );
           queueIndex = queueIndex - 1;
           
           neighbors = [ currentNode - 1, currentNode + 1, currentNode - sizeMatrix( 1), currentNode + sizeMatrix( 1 ) ];
           
           for neighbor = neighbors
               if neighbor > 0 && neighbor <= nElements && indicatorMatrix( neighbor ) == 1
                   queueIndex = queueIndex + 1;
                   queue( queueIndex ) = neighbor;
                   indicatorMatrix( neighbor ) = 0;
                   components( neighbor )  = currentComponent;
               end
           end
       end       
   end
   currentIndex = currentIndex + 1;
end