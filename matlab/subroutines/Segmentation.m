classdef Segmentation < handle
    properties
        roiNumber
        
        bestSegmentation
        bestLambda

        assignedLabel
        correctLabel
        
        correctSegmentation

        coordinates
        
        averageInt
        averageProcessed
        timeseries

        proposals
        lambdas
        rejectionSummary
    end
end