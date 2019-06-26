function [ y ] = getMean( s )
    y =  mean( s.preProcessFunc(), 3 );
