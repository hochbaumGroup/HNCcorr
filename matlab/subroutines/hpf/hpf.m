function [cuts, lambdas, stats, times] = hpf( arcmatrix, num_nodes, source_node, sink_node, lambda_range, rounding );                                             
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Hochbaum's Pseudo-flow (HPF) Algorithm for Parametric Minimimum Cut   %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The HPF algorithm for finding Minimum-cut in a graph is described in: %
% [1] D.S. Hochbaum, "The Pseudoflow algorithm: A new algorithm for the %
% maximum flow problem", Operations Research, 58(4):992-1009,2008.      %
%                                                                       %
% The algorithm was found to be fast in theory (see the above paper)    %
% and in practice (see:                                                 %
% [2] D.S. Hochbaum and B. Chandran, "A Computational Study of the      %
% Pseudoflow and Push-relabel Algorithms for the Maximum Flow Problem,  %
% Operations Research, 57(2):358-376, 2009.                             %
%                                                                       %
% and                                                                   %
%                                                                       %
% [3] B. Fishbain, D.S. Hochbaum, S. Mueller, "Competitive Analysis of  %
% Minimum-Cut Maximum Flow Algorithms in Vision Problems,               %
% arXiv:1007.4531v2 [cs.CV]                                             %
%                                                                       %
% The algorithm solves a parametric s-t minimum cut problem. The		%
% algorithm finds all breakpoints for which the source set of the		%
% minimum cut changes as a function of lambda in the range				%
% [lower bound, upper bound] by recursively concluding that the interval%
% contains 0, 1, or more breakpoints. If the interval contains more than%
% 1 breakpoint, then the interval is split into two interval, each of   %
% which contains at least one breakpoint.								%
%                                                                       %
% Parametric cut/flow problems allow for a linear function with input   %
% lambda on source or sink adjacent arcs. Arcs that are adjacent to     %
% source should be non-decreasing in lambda and sink adjacent arcs      %
% should be non-increasing in lambda. The algorithm is able to deal with%
% the reverse configuration (non-increasing on source adjacent arcs and %
% non-decreasing on sink adjacent arcs) by flipping source and sink and % 
% reversing the direction of the arcs.									%
%                                                                       %
% Usage:																%
% 1. Compile hpf.c with a C-compiler (e.g. gcc)                         %
% 2. To execute within bash environment:								%
%	 <name compiled hpf executable> <path input file> <path output file>%
%                                                                       %
% INPUT FILE                                                            %
% %%%%%%%%%%                                                            %
% The input file is assumed to be in a modified DIMACS format:	        %
% c <comment lines>                                                     %
% p <# nodes> <# arcs> <lower bound> <upper bound> <round if negative>  %
% n <source node> s                                                     %
% n <sink node> t                                                       %
% a <from-node> <to-node> <constant capacity> <lambda multiplier>		%
% where the following conditions are satisfied:                         %
% - Nodes are labeled 0 .. <# nodes> - 1								%
% - <lambda multiplier> is non-negative if <from-node> == <source node> %
%		and <to-node> != <sink-node>									%
% - <lambda multiplier> is non-positive if <from-node> != <source node> %
%		and <to-node> == <sink-node>									%
% - <lambda multiplier> is zero if <from-node> != <source node>         %
%		and <to-node> != <sink-node>								    %
% - <lambda multiplier> can take any value if						    %
%		<from-node> != <source node> and <to-node> != <sink-node>	    %
% - <round if negative> takes value 1 if the any negative capacity arc  %
%      should be rounded to 0, and it takes value 0 otherwise		    %
%                                                                       %
% OUTPUT FILE                                                           %
% %%%%%%%%%%%                                                           %
% The solver will generate the following output file:				    %
% t <time (in sec) read data> <time (in sec) initialize> <time		    %
%		(in sec) solve>												    %
% s <# arc scans> <# mergers> <# pushes> <# relabels > <# gap >		    %
% p <number of lambda intervals = k>								    %
% l <lambda upperbound interval 1> ... <lambda upperbound interval k>   %
% n <node-id> <sourceset indicator intval 1 > .. <indicator intval k>   %
%                                                                       %
% Set-up                                                                %
% %%%%%%                                                                %
% Uncompress the MatlabHPF.zip file into the Matlab's working directory %
% The zip file contains the following files:                            %
% hpf.c - source code                                                   %
% hpf.m - Matlab's help file                                            %
% hpf.mexmaci - The compiled code for Mac OS 10.0.5 (Intel)/ Matlab     %
%               7.6.0.324 (R2008a).                                     %
% hpf.mexw32  - The compiled code for Windows 7 / Matlab 7.11.0.584     %
%               (R2010b).                                               %
% demo_general - Short Matlab code that generates small network and     %
%                computes the minimum flow                              %
% demo_vision - Short Matlab code that loads a Multiview reconstruction %
%               vision problem (see: [3]) and computes its minimum cut. %
% gargoyle-smal.mat - The vision problem.                               %
%                                                                       %
% When using this code, please cite:                                    %
% References [1], [2] and [3] above and:                                %
% Q. Spaen, B. Fishbain and D.S. Hochbaum, "Hochbaum's Pseudo-flow C	%
% Implementation", http://riot.ieor.berkeley.edu/riot/Applications/     %
% Pseudoflow/maxflow.html                                               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[cuts, lambdas, stats, times] = hpfMatlab( arcmatrix, num_nodes, source_node, sink_node, lambda_range, rounding );