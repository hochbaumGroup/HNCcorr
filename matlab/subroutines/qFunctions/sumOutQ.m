function [ precomputedSum ] = sumOutQ( s )

% get precomputed sum
precomputedSum = full( sum( s.simMatrix( logical(s.test), : ), 2) )';