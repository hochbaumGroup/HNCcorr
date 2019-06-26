function [ accuracy, f1score, precision, recall, truePositives, falsePositives, trueNegatives, falseNegatives ] = evalSolution( s , lambdaPos )
global PLOT_DIR

% identify boundary pixels (boundary was artificially added) so they can be
% excluded for scoring
boundaryPixels = ones( s.effHeight, s.effWidth );
boundaryPixels( s.boundaryWidth + 1 : end - s.boundaryWidth, s.boundaryWidth + 1 : end - s.boundaryWidth ) = 0;
boundaryPixels = boundaryPixels(:);

assignedLabelTest = s.assignedLabels( logical(s.test), : );
trueLabelTest = repmat( s.labels( logical(s.test) ), 1, length( s.lambdas ) );
boundaryPixelsTest =  repmat( boundaryPixels( logical(s.test) ), 1, length( s.lambdas ) );

truePositives = sum( assignedLabelTest == trueLabelTest & assignedLabelTest == 1 & boundaryPixelsTest == 0 );
falsePositives = sum( assignedLabelTest ~= trueLabelTest & assignedLabelTest == 1 & boundaryPixelsTest == 0 );
trueNegatives = sum(  assignedLabelTest == trueLabelTest & assignedLabelTest == 0 & boundaryPixelsTest == 0 );
falseNegatives = sum(  assignedLabelTest ~= trueLabelTest & assignedLabelTest == 0 & boundaryPixelsTest == 0 );

accuracy = ( truePositives + trueNegatives ) ./ ( truePositives + trueNegatives + falsePositives + falseNegatives );
precision = truePositives ./ ( truePositives + falsePositives);
recall = truePositives ./ ( truePositives + falseNegatives );
f1score =  2 * ( precision .* recall ) ./ ( precision + recall );

fileNameBase = [ PLOT_DIR num2str(s.datasetID),'-', num2str(s.windowID),'-', num2str(s.preProcessID), '-', num2str(s.simSelectID), '-', num2str(s.simFunctionID), '-', num2str(s.qFunctionID),'-',num2str(s.algorithmID)  ];

if s.d.plotTuning || not( isempty( strfind( PLOT_DIR,'bestConfigs') ) ) || lambdaPos~=-1
    lb = 1;
    ub = length(s.lambdas);
    if lambdaPos ~=-1
        lb = lambdaPos;
        ub = lambdaPos;
    end
    for i = lb : ub
        close all;
        fig1 = figure('visible','off');
        pos_fig1 = [0 0 1000 1000];
        set(fig1,'Position',pos_fig1)
        set(fig1,'Units','Inches');
        posInch = get(fig1,'Position');
        set(fig1,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[posInch(3), posInch(4)])

        if s.configID == 1
            imagesc( s.meanMat );
            axis equal tight
            print( [ fileNameBase '-meanProcessed.pdf' ],'-dpdf');
            imagesc( mean( s.dataFile, 3 ) )
            axis equal tight
            print( [ fileNameBase '-meanUnprocessed.pdf' ],'-dpdf');
            
            dataFileVector = reshape(s.dataFile,s.effHeight*s.effWidth,[] );
            [~, ind] =  max( mean( dataFileVector( s.posSeeds,: ) ) );
            imagesc( s.dataFile(:,:,ind ) )
            axis equal tight
            print( [ fileNameBase '-meanMaxSeed.pdf' ],'-dpdf');
            

            imagesc( reshape( s.labels, s.effHeight, s.effWidth ) );
            axis equal tight
            print( [ fileNameBase '-TRUE.pdf' ],'-dpdf');
        end

        imagesc( reshape( s.assignedLabels(:,i), s.effHeight, s.effWidth ) );
        if s.lambda ~= -1
            title( strcat( 'lambda = ',num2str(s.lambda), ', alpha = ', num2str( s.alpha ), ' coordUpperLeft = [', strjoin( cellfun( @num2str, num2cell( s.coordinates( [ 1, 3 ] ) ),'UniformOutput',false), ' ' ), ']'  ) );
        elseif i == 1
            title( strcat( 'lambda = [0,', num2str( s.lambdas(i) ), '], alpha = ', num2str( s.alpha ), ' coordUpperLeft = [', strjoin( cellfun( @num2str, num2cell( s.coordinates( [ 1, 3 ] ) ),'UniformOutput',false), ' ' ), ']'  ) );
        else
            title( strcat( 'lambda = [', num2str(s.lambdas(i-1)),',', num2str( s.lambdas(i) ), '], alpha = ', num2str( s.alpha ), ' coordUpperLeft = [', strjoin( cellfun( @num2str, num2cell( s.coordinates( [ 1, 3 ] ) ),'UniformOutput',false), ' ' ), ']'  ) );
        end
        axis equal tight
        print( [ fileNameBase, '-',num2str(s.configID), '-', num2str(i), '.pdf' ],'-dpdf');
    end
end

end