classdef Solution < handle
    
    properties
        % quality of classification
        accuracy = '';
        precision = '';
        recall = '';
        f1score = '';
        truePositives = '';
        falsePositives = '';
        trueNegatives = '';
        falseNegatives = '';
        
        
        % running times
        cpuGetMean = 0;
        cpuSimSelect = 0;
        cpuSimFunction = 0;
        cpuqFunction = 0;
        cpuAlgorithm = 0;
        cpuClassify = 0;
        

        % ground truth json
        groundTruthJson = ''
        
        % functions
        datasetFunc = '';
        seedSelectionFunc = '';
        preProcessFunc = '';
        getDataForSeedSelection = '';
        simSelectFunc = '';
        simFunc = '';
        qFunc = '';
        algFunc = '';
        
        % dataset parameters
        neuronCover;
        dataFile = [];
        processedData = [];
        ROIs = [];
        datasetSize = [];
        nFrames = -1;
        nFramesEff = -1;
        boundaryWidth = -1;
        dataFileName;
        
        % window parameters
        windowSize = 31;
        coordinates = [];
        roiNumber = -1;
        effHeight = -1;
        effWidth = -1;
        
        % seed selection parameters
        seedCondition;
        seedSelectionThreshold;
        selectSeeds;
        posSeeds = [];
        negSeeds = [];
        test = [];
        labels = [];
        assignedLabels = [];
        seedMap;
        fileNameSeeds;
        numNegSeeds = 10;     
        pixelNotSegmented;        
        simListMean;
        gradientThreshold;
        
        percentilForSeeds;
        posSeedsSize;
        distNeg;
        maskSize
        maxShift = 4;
        detNegSeeds;
        
        potentialSeeds;
        potentialSeedIndex;
        
        bestSegmentation
        bestSegmentationIndex
        
        % preprocessing parameters
        framesSelect;
        coordinatesPreProcess = [0,0,0,0];
        
        % simSelect parameters & variables
        idxI = [];
        idxJ = [];
        
        neighborhoodSize = -1;
        
        % simFunc variables & parameters
        inBetweenVal = [];
        simVal = [];
        
        % qFunc variables
        inBetweenSimSource = [];
        alpha = -1;
        beta;
        
        % algorithm parameters & variables
        lambda = -1;
        lambdas;
        cuts;
        
        % mean properties
        meanMat = [];
		
		% sparse computation properties
		gridRes = -1;
        percCols = -1;
        percRows = -1;
		boxes;
		neighborBoxes;
		boxMapping;
		pcaScores;
        pcaCoeff;
        explainedVar;
        nPcaComps = -1;
        
        % parameters needed for SNC
        simMatrix = [];
        compressSimMatrix = [];
        sourceWeights = [];
        sinkWeights = [];
        precomputedSum = [];
        nTest = -1;
        
        % final results
        segmentations;
        numCells = 0;
        
        % average neuron size
        averageNeuronSize;
        maxNeuronPixels;
        minNeuronPixels;        
        fileNameBase;
    end
    methods
        function obj = Solution( fileNameBase, dataFileName, dataSize, nFrames, jsonFile, windowSize, minSize, avgSize, maxSize, posSeedSize, perc)
            % Setup configuration options  
            obj.fileNameBase = ['./HNCcorrResults/' fileNameBase];
            obj.dataFileName = dataFileName;
            obj.groundTruthJson = jsonFile;
            
            obj.boundaryWidth = 0;
            obj.datasetSize = dataSize;           
            obj.nFramesEff = nFrames;
            obj.windowSize = windowSize;
            obj.segmentations = {};
            obj.numCells = 0;
            obj.seedCondition = true;
            obj.roiNumber = 0;
            obj.minNeuronPixels = minSize;
            obj.averageNeuronSize = avgSize;
            obj.maxNeuronPixels = maxSize;
            obj.getDataForSeedSelection = @(varargin) getRawData(obj,varargin);
            %this.ROIs = jsonToROIsConverter( this.datasetAbbr, this.datasetSize );

            try
                disp(obj.dataFileName);
                h5disp( obj.dataFileName );
            catch
                error('Invalid h5 file');
            end
            
            if not( strcmp( obj.groundTruthJson,'') )
                obj.ROIs = jsonToROIsConverter( jsonFile, obj.datasetSize, windowSize );
            end
            
            if ~isempty(obj.ROIs)
                obj.neuronCover = computeNeuronCover( obj );
            end
            
            % Seed Selection Setup
            obj.distNeg = 0.058*obj.windowSize^1.5;
            obj.maskSize = 1; %Configurable parameter or fixed one?
            obj.posSeedsSize = posSeedSize;
            obj.percentilForSeeds = perc;
            obj.detNegSeeds = 0;
            obj.selectSeeds = @(x) fiveByFiveAllPixels( obj, obj.distNeg );
            %this.spikes = containers.Map;
            
            % Preprocessing function setup to correlation space
            obj.preProcessFunc = @(varargin) preprocessCorrelation(obj,varargin);
            
            % Similarity selection function setup to sparse computation
            % with default values
            obj.simSelectFunc = @(void) sparseComputation( obj );
            obj.gridRes = 25;
            obj.nPcaComps = 3;
            obj.percRows = 0.01;
            obj.percCols = 1;
            
            %Similarity function setup to Euclidian distance
            obj.simFunc = @( idxI, idxJ, recomputeFlag, duplicate ) computeNormSim2( obj, idxI, idxJ, recomputeFlag , duplicate );
            
            %Similarity Q function setup to sumTimesSource
            obj.qFunc = @( recomputeSimFlag ) sumOutQ( obj );
            
            %Algorithm changed to Normalized cut
            obj.algFunc = @( x ) SNC( obj );
        end
                
        function changeWindow( this, varargin )
            % check if ID or window specification is given
            this.roiNumber = varargin{1};            
            
            %coordinates are assigned in dataSeedSelection, This
            %coordinates come from the last execution of dataSeedSelection.
            this.effWidth = this.coordinates( 4 ) - this.coordinates( 3 ) + 1 + 2 * this.boundaryWidth;
            this.effHeight = this.coordinates( 2 ) - this.coordinates( 1 ) + 1 + 2 * this.boundaryWidth;
            
            if this.boundaryWidth == 0
                this.dataFile = h5read( this.dataFileName, '/data', [ this.coordinates(1), this.coordinates(3), 1 ], ...
                    [ this.coordinates( 2 ) - this.coordinates( 1 ) + 1, this.coordinates( 4 ) - this.coordinates( 3 ) + 1, this.nFramesEff ] );
            else
                this.dataFile = zeros( this.effHeight, this.effWidth, this.nFramesEff );
                this.dataFile( this.boundaryWidth + 1 : end - this.boundaryWidth, this.boundaryWidth + 1 : end - this.boundaryWidth, : ) = h5read( this.dataFileName, '/data', ...
                    [ this.coordinates(1), this.coordinates(3), 1 ], ...
                    [ this.coordinates( 2 ) - this.coordinates( 1 ) + 1, this.coordinates( 4 ) - this.coordinates( 3 ) + 1, this.nFramesEff ] );
            end
            this.datasetFunc = @(x) this.dataFile;
        end
                                          
        function saveSeedSelectionPlots( this )
            h = figure('visible','off');
            imagesc(reshape(this.simListMean,this.datasetSize(1),this.datasetSize(2)));
            fileName = [ 'seedSelectionPlots/simListMean-' this.fileNameBase '.pdf'];
            saveas(gcf,fileName,'pdf');
        end                
    end
end
