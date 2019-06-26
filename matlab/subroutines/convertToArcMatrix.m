function [ arcMatrix ] = convertToArcMatrix( constantCapacities, sourceMultipliers, sinkMultipliers, source, sink)
if ~issparse( constantCapacities )
    error( 'constantCapacities matrix has to be sparse. Use sparse() function' )
end
nNodes = size( constantCapacities, 1 );
if nNodes ~= size(sourceMultipliers,1)
    error('sourceMultipliers has to be an n x 1 vector, where n is the number of nodes');
end
if nNodes ~= size(sinkMultipliers,1)
    error('sinkMultipliers has to be an n x 1 vector, where n is the number of nodes');
end

% find number of nodes and arcs
nNodes = size( constantCapacities, 1 );
nArcs = nnz( constantCapacities );

notSourceProcessed = ones( nNodes, 1 );
notSinkProcessed = ones( nNodes, 1 );

[in,out,valueArc ] = find( constantCapacities );

sourceAdjacent = in == source;
sinkAdjacent = out == sink;

if max( sourceAdjacent & sinkAdjacent ) == 1
    error('Please remove any arcs from source to sink');
end

sourceAdjacentMultiplier = sourceMultipliers( out( sourceAdjacent ) );
notSourceProcessed( out( sourceAdjacent ) ) = 0;
nSourceAdjacent = length( sourceAdjacentMultiplier );

notSourceProcessed = ( sourceMultipliers .* notSourceProcessed ) > 0;
nNotSourceProcessed = sum( notSourceProcessed );

sinkAdjacentMultiplier = sinkMultipliers( in( sinkAdjacent ) );
notSinkProcessed( in( sinkAdjacent ) ) = 0;
nSinkAdjacent = length( sinkAdjacentMultiplier );

notSinkProcessed = ( sinkMultipliers .* notSinkProcessed ) > 0;
nNotSinkProcessed = sum( notSinkProcessed );

notAdjacent = ~ ( sourceAdjacent | sinkAdjacent );
nNotAdjacent = sum( notAdjacent);

nArcsProcessed = 0;
arcMatrix = zeros( nArcs + nNotSinkProcessed + nNotSourceProcessed, 4 );

arcMatrix( nArcsProcessed + 1 : nArcsProcessed + nSourceAdjacent, : ) = [ source * ones( nSourceAdjacent, 1 ), out( sourceAdjacent), valueArc( sourceAdjacent), sourceAdjacentMultiplier ];
nArcsProcessed = nArcsProcessed + nSourceAdjacent;

arcMatrix( nArcsProcessed + 1 : nArcsProcessed + nSinkAdjacent, : ) = [ in( sinkAdjacent), sink * ones( nSinkAdjacent, 1 ) , valueArc( sinkAdjacent), sinkAdjacentMultiplier ];
nArcsProcessed = nArcsProcessed + nSinkAdjacent;

arcMatrix( nArcsProcessed + 1 : nArcsProcessed + nNotAdjacent, : ) = [ in( notAdjacent), out( notAdjacent ) , valueArc( notAdjacent), zeros( nNotAdjacent, 1 ) ];
nArcsProcessed = nArcsProcessed + nNotAdjacent;

arcMatrix( nArcsProcessed + 1 : nArcsProcessed + nNotSourceProcessed, : ) = [ source * ones( nNotSourceProcessed, 1 ), find( notSourceProcessed ), zeros( nNotSourceProcessed, 1 ), sourceMultipliers( notSourceProcessed ) ];
nArcsProcessed = nArcsProcessed + nNotSourceProcessed;

arcMatrix( nArcsProcessed + 1 : nArcsProcessed + nNotSinkProcessed, : ) = [ find( notSinkProcessed ), sink * ones( nNotSinkProcessed, 1 ), zeros( nNotSinkProcessed, 1 ), sinkMultipliers( notSinkProcessed ) ];
nArcsProcessed = nArcsProcessed + nNotSinkProcessed;

assert( nArcsProcessed == nArcs + nNotSinkProcessed + nNotSourceProcessed);