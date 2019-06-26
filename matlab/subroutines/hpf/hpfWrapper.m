function [ cuts, lambdas, times, stats ] = hpfWrapper( constantCapacities, sourceMultipliers, sinkMultipliers, source, sink, lambdaLow, lambdaHigh, roundNegCapacity )
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

sourceProcessed = zeros( nNodes, 1 );
sinkProcessed = zeros( nNodes, 1 );

[in,out,valueArc ] = find( constantCapacities );

sourceSink = sprintf( 'n %u s\nn %u t\n',source - 1, sink - 1 );

arcs = '';

% process arcs
for i = 1 : nArcs
    if in(i) == source
        multiplier = sourceMultipliers( out(i) );
        sourceProcessed( out(i) ) = 1;
    elseif out(i) == sink
        multiplier = sinkMultipliers( in(i) );
        sinkProcessed( in(i) ) = 1;
    else
        multiplier = 0;
    end
    arcs = [ arcs sprintf('a %u %u %.12f %.12f\n',in(i)-1,out(i)-1,valueArc(i),multiplier)];
end

% process additional source / sink adjacent arcs
for i = 1 : nNodes
    if ( sourceMultipliers(i) > 0 && sourceProcessed(i) == 0 )
        arcs = [ arcs sprintf('a %u %u %.12f %.12f\n',source-1,i-1,0,sourceMultipliers(i))];
        nArcs = nArcs + 1;
    end
    if ( sinkMultipliers(i) > 0 && sinkProcessed(i) == 0 )
        arcs = [ arcs sprintf('a %u %u %.12f %.12f\n',i-1,sink-1,0,sinkMultipliers(i))];
        nArcs = nArcs + 1;
    end
end

% remove trailing new line
arcs = arcs( 1: end -2);

% print numNodes, numArcs, and lambda values to string
problem = sprintf( 'p %d %d %.12f %.12f %u\n', nNodes, nArcs, lambdaLow, lambdaHigh, roundNegCapacity );

% open input file
f = fopen( 'inputHPF.txt', 'w');

% print to file
fprintf( f, '%s%s%s', problem, sourceSink, arcs );

% close input file
fclose( f );

% call HPF solver
if system( 'subroutines\hpf\hpf.exe inputHPF.txt outputHPF.txt' ) == 0
    error('Solver crashed');
end

% open output file
f = fopen( 'outputHPF.txt', 'r' );

% read line by line
tline = fgets(f);
while ischar(tline)
    lineCell =  strsplit( tline );
    type = lineCell{1};
    values = cellfun(@str2num,lineCell(2:end-1) );
    
    % process output
    if type == 't'
        times = values;
    elseif type == 's'
        stats = values;
    elseif type == 'p'
        numBreakpoints = values;
        cuts = zeros(nNodes, numBreakpoints );
    elseif type == 'l'
        lambdas = values;
    elseif type == 'n'
        index = values(1) + 1;
        cuts( index, : ) = values( 2 : end ); 
    end
    % get next line
    tline = fgets(f);
end

% close output file
fclose( f );
end