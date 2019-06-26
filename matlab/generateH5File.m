function generateH5File(fileNameBase, rawdata_dir, dataSize, nFiles, chunkSize, binSize)

    nFramesEff = ceil(nFiles/binSize);
    tiffFileNameBase = [rawdata_dir '/image'];
    tiffInfo = initDataStream(tiffFileNameBase, nFiles);
    system('mkdir h5Files');
    h5FileName = ['./h5Files/' fileNameBase '_' int2str(binSize) '.h5'];
    h5create(h5FileName,'/data',[dataSize(1) dataSize(2) nFramesEff],'ChunkSize',[chunkSize, chunkSize, nFramesEff]);
    cont=1;
    for i = 0:binSize:nFiles-binSize
        h5write( h5FileName,'/data', dataStream( i,  tiffFileNameBase, binSize, nFiles, dataSize, tiffInfo), [1, 1, cont ], [ dataSize(1) dataSize(2), 1 ] );
        cont = cont+1;
    end
end

function [ tiffInfo ] = initDataStream(baseFileName, nFiles)
    tiffInfo = cell( nFiles, 1 );
    for i = 0 : nFiles-1
        fileName = strcat( baseFileName , num2str( i, '%05d' ), '.tiff' );
        tiffInfo{ i+1 } = imfinfo( fileName );
    end
end

function [ meanData ] = dataStream( fr, baseFileName, binSize, nFiles, dataSize, tiffInfo)
    % binned and averaged over binSize frames.
    nFrames = binSize;
    if(fr + binSize >= nFiles)
        nFrames = nFiles-fr;
    end
    FinalImage = zeros(dataSize(1), dataSize(2), nFrames);
    for i = fr:fr + binSize - 1 %read files from fr to fr+binSize
       if( i == nFiles)
           break;
       end
       InfoImage= tiffInfo{i+1};
       fileName = strcat( baseFileName , num2str( i, '%05d' ), '.tiff' );
       args = struct( 'index', 1, 'pixelregion', struct([]), 'info', InfoImage, 'filename', fileName, 'offset', 0 );%tiff containing only one image
       FinalImage(:,:,i-fr+1) = rtifc(args);
    end
    noisyData = double(FinalImage);
    PreMeanCell = mat2cell( noisyData ,dataSize(1),dataSize(2), repmat( nFrames, 1, 1 ) );
    clear noisyData;

    % average frame intensities for each bin
    MeanCell = cellfun( @(x) mean(x,3), PreMeanCell, 'UniformOutput',false );
    clear PreMeanCell;

    % convert back to matrix
    meanData = cell2mat( MeanCell);
end
