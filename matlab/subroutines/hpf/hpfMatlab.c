/*************************************************************************
 * Hochbaum's Pseudo-flow (HPF) Algorithm for Parametric Minimimum Cut   *
 * ***********************************************************************
 * The HPF algorithm for finding Minimum-cut in a graph is described in: *
 * [1] D.S. Hochbaum, "The Pseudoflow algorithm: A new algorithm for the *
 * maximum flow problem", Operations Research, 58(4):992-1009,2008.      *
 *                                                                       *
 * The algorithm was found to be fast in theory (see the above paper)    *
 * and in practice (see:                                                 *
 * [2] D.S. Hochbaum and B. Chandran, "A Computational Study of the      *
 * Pseudoflow and Push-relabel Algorithms for the Maximum Flow Problem,  *
 * Operations Research, 57(2):358-376, 2009.                             *
 *                                                                       *
 * and                                                                   *
 *                                                                       *
 * [3] B. Fishbain, D.S. Hochbaum, S. Mueller, "Competitive Analysis of  *
 * Minimum-Cut Maximum Flow Algorithms in Vision Problems,               *
 * arXiv:1007.4531v2 [cs.CV]                                             *
 *                                                                       *
 * The algorithm solves a parametric s-t minimum cut problem. The		 *
 * algorithm finds all breakpoints for which the source set of the		 *
 * minimum cut changes as a function of lambda in the range				 *
 * [lower bound, upper bound] by recursively concluding that the interval  *
 * contains 0, 1, or more breakpoints. If the interval contains more than*
 * 1 breakpoint, then the interval is split into two interval, each of   *
 * which contains at least one breakpoint.								 *
 *                                                                       *
 * Parametric cut/flow problems allow for a linear function with input   *
 * lambda on source or sink adjacent arcs. Arcs that are adjacent to	 *
 * source should be non-decreasing in lambda and sink adjacent arcs		 *
 * should be non-increasing in lambda. The algorithm is able to deal with*
 * the reverse configuration (non-increasing on source adjacent arcs and *
 * non-decreasing on sink adjacent arcs) by flipping source and sink and * 
 * reversing the direction of the arcs.									 *
 *                                                                       *
 * Usage:																 *
 * 1. Compile hpf.c with a C-compiler (e.g. gcc)						 *
 * 2. To execute within bash environment:								 *
 *	 <name compiled hpf executable> <path input file> <path output file> *
 *                                                                       *
 * INPUT FILE                                                            *
 * **********                                                            *
 * The input file is assumed to be in a modified DIMACS format:	         *
 * c <comment lines>													 *
 * p <# nodes> <# arcs> <lower bound> <upper bound> <round if negative>  *
 * n <source node> s													 *
 * n <sink node> t														 *
 * a <from-node> <to-node> <constant capacity> <lambda multiplier>		 *
 * where the following conditions are satisfied:						 *
 * - Nodes are labeled 0 .. <# nodes> - 1								 *
 * - <lambda multiplier> is non-negative if <from-node> == <source node> *
 *		and <to-node> != <sink-node>									 *
 * - <lambda multiplier> is non-positive if <from-node> != <source node> *
 *		and <to-node> == <sink-node>									 *
 * - <lambda multiplier> is zero if <from-node> != <source node>		 *
 *		and <to-node> != <sink-node>									 *
 * - <lambda multiplier> can take any value if							 *
 *		<from-node> != <source node> and <to-node> != <sink-node>		 *
 * - <round if negative> takes value 1 if the any negative capacity arc  *
 *		should be rounded to 0, and it takes value 0 otherwise			 *
 *                                                                       *
 * OUTPUT FILE                                                           *
 * ***********                                                           *
 * The solver will generate the following output file:					 *
 * t <time (in sec) read data> <time (in sec) initialize> <time			 *
		(in sec) solve>													 *
 * s <# arc scans> <# mergers> <# pushes> <# relabels > <# gap >		 *
 * p <number of lambda intervals = k>									 *
 * l <lambda upperbound interval 1> ... <lambda upperbound interval k>   *
 * n <node-id> <sourceset indicator intval 1 > .. <indicator intval k>   *
 *                                                                       *
 * Set-up                                                                *
 * ******                                                                *
 * Uncompress the MatlabHPF.zip file into the Matlab's working directory *
 * The zip file contains the following files:                            *
 * hpf.c - source code                                                   *
 * hpf.m - Matlab's help file                                            *
 * hpf.mexmaci - The compiled code for Mac OS 10.0.5 (Intel)/ Matlab     *
 *               7.6.0.324 (R2008a).                                     *
 * hpf.mexw32  - The compiled code for Windows 7 / Matlab 7.11.0.584     *
 *               (R2010b).                                               *
 * demo_general - Short Matlab code that generates small network and     *
 *                computes the minimum flow                              *
 * demo_vision - Short Matlab code that loads a Multiview reconstruction *
 *               vision problem (see: [3]) and computes its minimum cut. *
 * gargoyle-smal.mat - The vision problem.                               *
 *                                                                       *
 * When using this code, please cite:                                    *
 * References [1], [2] and [3] above and:                                *
 * Q. Spaen, B. Fishbain and D.S. Hochbaum, "Hochbaum's Pseudo-flow C	 *
 * Implementation", http://riot.ieor.berkeley.edu/riot/Applications/     *
 * Pseudoflow/maxflow.html                                               *
 *************************************************************************/

#define _CRTDBG_MAP_ALLOC
#include "mex.h"
#include "matrix.h"
#include "stdio.h"
//#include <sys/time.h>
//#include <sys/resource.h>
#include "stdlib.h"
#include "time.h"
//#include <unistd.h>

/*************************************************************************
Definitions
*************************************************************************/
#define  MAX_LEVELS  300
#define VERSION 3.3

typedef unsigned int uint;
typedef long int lint;
typedef long long int llint;
typedef unsigned long long int ullint;

typedef struct Arc
	{
		struct Node *from;
		struct Node *to;
		double flow;
		double capacity;
		double constant;
		double multiplier;
		uint direction;
	} Arc;

typedef struct Node
	{
		uint visited;
		uint numAdjacent;
		uint number;
		int originalIndex;
		uint label;
		double excess;
		struct Node *parent;
		struct Node *childList;
		struct Node *nextScan;
		uint numOutOfTree;
		Arc **outOfTree;
		uint nextArc;
		Arc *arcToParent;
		struct Node *next;
	} Node;

typedef struct CutProblem
{
	uint numNodesInList;
	uint numSourceSet;
	uint numSinkSet;
	uint numArcs;
	uint solved;
	double lambdaValue;
	Arc *arcList;
	Node *nodeList;
	double cutValue;
	double cutMultiplier;
	double cutConstant;
	Node *sourceSet;
	Node *sinkSet;
	uint *optimalSourceSetIndicator;
} CutProblem;

typedef struct Root
{
	Node *start;
	Node *end;
} Root;

typedef struct Breakpoint
{
	double lambdaValue;
	uint* sourceSetIndicator;
	struct Breakpoint *next;
} Breakpoint;

#ifndef TRUE
#define TRUE (1)
#endif

#ifndef FALSE
#define FALSE (0)
#endif

/*************************************************************************
Global variables
*************************************************************************/
// tolerance for denominator == 0
static double TOL = 1E-8;
static uint numNodes = 0;
static uint numArcs = 0;
static uint numNodesSuper = 0;
static uint numArcsSuper = 0;
static uint source;
static uint sink;
static uint highestStrongLabel = 1;

static uint numArcScans = 0;
static uint numPushes = 0;
static uint numMergers = 0;
static uint numRelabels = 0;
static uint numGaps = 0;

static Node *nodesList = NULL;
static Root *strongRoots = NULL;
static uint *labelCount = NULL;
static Arc *arcList = NULL;
static Node *nodeListSuper = NULL;
static Arc *arcListSuper = NULL;
static uint lowestPositiveExcessNode = 0;

static Breakpoint *lastBreakpoint = NULL;
static Breakpoint *firstBreakpoint = NULL;

static uint useParametricCut = 1;
static uint roundNegativeCapacity = 0;

static double LAMBDA_LOW;
static double LAMBDA_HIGH;

/* Input Arguments */
#define	ARC_MATRIX		prhs[0]
#define	NUM_NODES		prhs[1]
#define SOURCE			prhs[2]
#define SINK			prhs[3]
#define	LAMBDA_RANGE	prhs[4]
#define	ROUNDING		prhs[5]

/* Output Arguments */
#define	CUTS	plhs[0]
#define	LAMBDAS plhs[1]

double dabs(double value)
{
	if (value >= 0)
		return value;
	else return -value;
}

int isFlow(double flow)
/*************************************************************************
isFlow: We set a threshhold. If the flow value is below the threshhold, we
take it as no flow. Otherwise we take it as a flow
*************************************************************************/
{
	if (flow > 0)
		return 1;
	else return 0;
}

int isExcess(double excess)
/*************************************************************************
isExcess: We set a threshhold. If the absolute value of the excess is within
the threshold, then we take it as nothing. Otherwise we will return the sign
of the excess (deficit if negative).
*************************************************************************/
{
	if (excess < 0)
		return -1;
	else if (excess > 0)
		return 1;
	else return 0;
}

static void createOutOfTree (Node *nd)
{
/*************************************************************************
createOutOfTree
*************************************************************************/
	if (nd->numAdjacent)
	{
		if ((nd->outOfTree = (Arc **) mxMalloc(nd->numAdjacent * sizeof (Arc *))) == NULL)
		{
			printf("Out of memory\n");
			exit(0);
		}
	}
}

static void initializeArc (Arc *ac)
{
/*************************************************************************
initializeArc
*************************************************************************/
	ac->from = NULL;
	ac->to = NULL;
	ac->capacity = 0.0;
	ac->flow = 0.0;
	ac->direction = 1;
	ac->constant = 0.0;
	ac->multiplier = 0.0;
}

static void liftAll (Node *rootNode)
{
/*************************************************************************
liftAll
*************************************************************************/
	Node *temp, *current=rootNode;

	current->nextScan = current->childList;

	-- labelCount[current->label];
	current->label = numNodes;

	for ( ; (current); current = current->parent)
	{
		while (current->nextScan)
		{
			temp = current->nextScan;
			current->nextScan = current->nextScan->next;
			current = temp;
			current->nextScan = current->childList;

			-- labelCount[current->label];
			current->label = numNodes;
		}
	}
}
static void addOutOfTreeNode (Node *n, Arc *out)
{
/*************************************************************************
addOutOfTreeNode
*************************************************************************/
	n->outOfTree[n->numOutOfTree] = out;
	++ n->numOutOfTree;
}

static void addToStrongBucket (Node *newRoot, Root *rootBucket)
{
/*************************************************************************
addToStrongBucket
*************************************************************************/
	if (rootBucket->start)
	{
		rootBucket->end->next = newRoot;
		rootBucket->end = newRoot;
		newRoot->next = NULL;
	}
	else
	{
		rootBucket->start = newRoot;
		rootBucket->end = newRoot;
		newRoot->next = NULL;
	}
}

static __inline int addRelationship (Node *newParent, Node *child)
{
/*************************************************************************
addRelationship
*************************************************************************/
	child->parent = newParent;
	child->next = newParent->childList;
	newParent->childList = child;

	return 0;
}

static __inline void breakRelationship (Node *oldParent, Node *child)
{
/*************************************************************************
breakRelationship
*************************************************************************/
	Node *current;

	child->parent = NULL;

	if (oldParent->childList == child)
	{
		oldParent->childList = child->next;
		child->next = NULL;
		return;
	}

	for (current = oldParent->childList; (current->next != child); current = current->next);

	current->next = child->next;
	child->next = NULL;
}

static void merge (Node *parent, Node *child, Arc *newArc)
{
/*************************************************************************
merge
*************************************************************************/
	Arc *oldArc;
	Node *current = child, *oldParent, *newParent = parent;

	++ numMergers;

	while (current->parent)
	{
		oldArc = current->arcToParent;
		current->arcToParent = newArc;
		oldParent = current->parent;
		breakRelationship (oldParent, current);
		addRelationship (newParent, current);
		newParent = current;
		current = oldParent;
		newArc = oldArc;
		newArc->direction = 1 - newArc->direction;
	}

	current->arcToParent = newArc;
	addRelationship (newParent, current);
}


static __inline void pushUpward (Arc *currentArc, Node *child, Node *parent, const double resCap)
{
/*************************************************************************
pushUpward
*************************************************************************/
	++ numPushes;

	if (isExcess(resCap-child->excess) >= 0)//(/*(int)*/resCap >= child->excess)
	{
		parent->excess += child->excess;
		currentArc->flow += child->excess;
		child->excess = 0;
		return;
	}

	currentArc->direction = 0;
	parent->excess += resCap;
	child->excess -= resCap;
	currentArc->flow = currentArc->capacity;
	parent->outOfTree[parent->numOutOfTree] = currentArc;
	++ parent->numOutOfTree;
	breakRelationship (parent, child);

	addToStrongBucket (child, &strongRoots[child->label]);
}


static __inline void pushDownward (Arc *currentArc, Node *child, Node *parent, double flow)
{
/*************************************************************************
pushDownward
*************************************************************************/
	++ numPushes;

	if (isExcess(flow - child->excess) >= 0)//(/*(int)*/flow >= child->excess)
	{
		parent->excess += child->excess;
		currentArc->flow -= child->excess;
		child->excess = 0;
		return;
	}

	currentArc->direction = 1;
	child->excess -= flow;
	parent->excess += flow;
	currentArc->flow = 0;
	parent->outOfTree[parent->numOutOfTree] = currentArc;
	++ parent->numOutOfTree;
	breakRelationship (parent, child);

	addToStrongBucket (child, &strongRoots[child->label]);
}

static void printCutProblem(CutProblem *p){
  printf("numNodes: %u\n " ,p->numNodesInList);
  printf("numSource %u\n" ,p->numSourceSet);
  printf("numSink: %u\n" ,p->numSinkSet);
  printf("numArcs: %u\n" ,p->numArcs);
  printf("solved: %u\n" ,p->solved);
  printf("lambda:%.12lf\n" ,p->lambdaValue);
  int i;
  for(i=0;i<numArcs;++i){
    printf("[%d,%d](%.12lf,%.12lf,%.12lf)\n",p->arcList[i].from->originalIndex,p->arcList[i].to->originalIndex,p->arcList[i].capacity,p->arcList[i].constant,p->arcList[i].multiplier);
  }
  printf("\n");
  //printArcListInfo(arcList);
  //printNodeListInfo(nodeList);
  printf("%.12lf " ,p->cutValue);
  printf("%.12lf " ,p->cutMultiplier);
  printf("%.12lf " ,p->cutConstant);
  //printNodeListInfo(sourceSet);
  //printSinkListInfo(sinkSet);
  for(i=0;i<numNodesSuper;++i){
    printf("%u ",p->optimalSourceSetIndicator[i]);
  }
  printf("\n");
}

static void pushExcess (Node *strongRoot)
{
/*************************************************************************
pushExcess
*************************************************************************/
	Node *current, *parent;
	Arc *arcToParent;
	/*int*/double prevEx=1;

	for (current = strongRoot; (isExcess(current->excess) && current->parent); current = parent)
	{
		parent = current->parent;
		prevEx = parent->excess;

		arcToParent = current->arcToParent;

		if (arcToParent->direction)
		{
			pushUpward (arcToParent, current, parent, (arcToParent->capacity - arcToParent->flow));
		}
		else
		{
			pushDownward (arcToParent, current, parent, arcToParent->flow);
		}
	}

	if ((isExcess(current->excess) > 0) && (isExcess(prevEx) <= 0))
	{
		addToStrongBucket (current, &strongRoots[current->label]);
	}
}


static Arc * findWeakNode (Node *strongNode, Node **weakNode)
{
/*************************************************************************
findWeakNode
*************************************************************************/
	uint i, size;
	Arc *out;

	size = strongNode->numOutOfTree;

	for (i=strongNode->nextArc; i<size; ++i)
	{
		++ numArcScans;
		if (strongNode->outOfTree[i]->to->label == (highestStrongLabel-1))
		{
			strongNode->nextArc = i;
			out = strongNode->outOfTree[i];
			(*weakNode) = out->to;
			-- strongNode->numOutOfTree;
			strongNode->outOfTree[i] = strongNode->outOfTree[strongNode->numOutOfTree];
			return (out);
		} else if (strongNode->outOfTree[i]->from->label == (highestStrongLabel-1)) {
			strongNode->nextArc = i;
			out = strongNode->outOfTree[i];
			(*weakNode) = out->from;
			-- strongNode->numOutOfTree;
			strongNode->outOfTree[i] = strongNode->outOfTree[strongNode->numOutOfTree];
			return (out);
		}
	}

	strongNode->nextArc = strongNode->numOutOfTree;

	return NULL;
}


static void checkChildren (Node *curNode)
{
/*************************************************************************
checkChildren
*************************************************************************/
	for ( ; (curNode->nextScan); curNode->nextScan = curNode->nextScan->next)
	{
		if (curNode->nextScan->label == curNode->label)
		{
			return;
		}

	}

	-- labelCount[curNode->label];
	++	curNode->label;
	++ labelCount[curNode->label];

	++numRelabels;

	curNode->nextArc = 0;
}


static void simpleInitialization (void)
{
/*************************************************************************
simpleInitialization
*************************************************************************/
	uint i, size;
	Arc *tempArc;

	size = nodesList[source].numOutOfTree;
	for (i=0; i<size; ++i) // Saturating source adjacent nodes
	{
		tempArc = nodesList[source].outOfTree[i];
		tempArc->flow = tempArc->capacity;
		tempArc->to->excess += tempArc->capacity;
	}

	size = nodesList[sink].numOutOfTree;
	for (i=0; i<size; ++i) // Pushing maximum flow on sink adjacent nodes
	{
		tempArc = nodesList[sink].outOfTree[i];
		tempArc->flow = tempArc->capacity;
		tempArc->from->excess -= tempArc->capacity;
	}

	nodesList[source].excess = 0; // zeroing source excess
	nodesList[sink].excess = 0;	// zeroing sink excess

	for (i=0; i<numNodes; ++i)
	{
		if (isExcess(nodesList[i].excess) > 0)
		{
		    nodesList[i].label = 1;
			++ labelCount[1];

			addToStrongBucket (&nodesList[i], &strongRoots[1]);
		}
	}

	nodesList[source].label = numNodes;	// Set the source label to n
	nodesList[sink].label = 0;			// set the sink label to 0
	labelCount[0] = (numNodes - 2) - labelCount[1];
}


static Node* getHighestStrongRoot (void)
{
/*************************************************************************
getHighestStrongRoot
*************************************************************************/
	uint i;
	Node *strongRoot;

	for (i=highestStrongLabel; i>0; --i)
	{
		if (strongRoots[i].start)
		{
			highestStrongLabel = i;
			if (labelCount[i-1])
			{
				strongRoot = strongRoots[i].start;
				strongRoots[i].start = strongRoot->next;
				strongRoot->next = NULL;
				return strongRoot;
			}

			while (strongRoots[i].start)
			{
				++ numGaps;

				strongRoot = strongRoots[i].start;
				strongRoots[i].start = strongRoot->next;
				liftAll (strongRoot);
			}
		}
	}

	if (!strongRoots[0].start)
	{
		return NULL;
	}

	while (strongRoots[0].start)
	{
		strongRoot = strongRoots[0].start;
		strongRoots[0].start = strongRoot->next;
		strongRoot->label = 1;
		-- labelCount[0];
		++ labelCount[1];

		++ numRelabels;

		addToStrongBucket (strongRoot, &strongRoots[strongRoot->label]);
	}

	highestStrongLabel = 1;

	strongRoot = strongRoots[1].start;
	strongRoots[1].start = strongRoot->next;
	strongRoot->next = NULL;

	return strongRoot;
}



static void initializeRoot (Root *rt)
{
/*************************************************************************
initializeRoot
*************************************************************************/
	rt->start = NULL;
	rt->end = NULL;
}


static void initializeNode (Node *nd, const uint n)
{
/*************************************************************************
initializeNode
*************************************************************************/
	nd->label = 0;
	nd->excess = 0.0;
	nd->parent = NULL;
	nd->childList = NULL;
	nd->nextScan = NULL;
	nd->nextArc = 0;
	nd->numOutOfTree = 0;
	nd->arcToParent = NULL;
	nd->next = NULL;
	nd->visited = 0;
	nd->numAdjacent = 0;
	nd->number = n;
	nd->originalIndex = -10; 
	nd->outOfTree = NULL;
}

static void destroyBreakpoint(Breakpoint *currentBreakpoint)
/*************************************************************************
destroyBreakpoint - Removes breakpoint and subsequent ones
*************************************************************************/
{
	if (currentBreakpoint == NULL)
	{
		return;
	}
	/* free sourceset indicator */
	mxFree(currentBreakpoint->sourceSetIndicator);
	currentBreakpoint->sourceSetIndicator = NULL;

	/* iterate through breakpoint list */
	destroyBreakpoint(currentBreakpoint->next);
	currentBreakpoint->next = NULL;

	/* free breakpoint */
	mxFree(currentBreakpoint);
	currentBreakpoint == NULL;

}

static void freeRoot (Root *rt)
{
/*************************************************************************
freeRoot
*************************************************************************/
	rt->start = NULL;
	rt->end = NULL;
}

static void freeMemoryComplete(void)
/*************************************************************************
freeMemoryComplete
*************************************************************************/
{
	/* destroy breakpoints */
	destroyBreakpoint(firstBreakpoint);
	firstBreakpoint = NULL;

	mxFree(nodeListSuper);
	nodeListSuper = NULL;
	mxFree(arcListSuper);
	arcListSuper = NULL;
}

static void freeMemorySolve (void)
{
/*************************************************************************
freeMemorySolve
*************************************************************************/
	uint i;

	for (i=0; i<numNodes; ++i)
	{
		freeRoot (&strongRoots[i]);
	}

	mxFree(strongRoots);
	strongRoots = NULL;

	for (i=0; i<numNodes; ++i)
	{
		if (nodesList[i].outOfTree)
		{
			mxFree(nodesList[i].outOfTree);
			nodesList[i].outOfTree = NULL;
		}
	}

	mxFree(labelCount);
	labelCount = NULL;
}

static void processRoot (Node *strongRoot)
{
/*************************************************************************
processRoot
*************************************************************************/
	Node *temp, *strongNode = strongRoot, *weakNode;
	Arc *out;

	strongRoot->nextScan = strongRoot->childList;

	if ((out = findWeakNode (strongRoot, &weakNode)))
	{
		merge (weakNode, strongNode, out);
		pushExcess (strongRoot);
		return;
	}

	checkChildren (strongRoot);

	while (strongNode)
	{
		while (strongNode->nextScan)
		{
			temp = strongNode->nextScan;
			strongNode->nextScan = strongNode->nextScan->next;
			strongNode = temp;
			strongNode->nextScan = strongNode->childList;

			if ((out = findWeakNode (strongNode, &weakNode)))
			{
				merge (weakNode, strongNode, out);
				pushExcess (strongRoot);
				return;
			}

			checkChildren (strongNode);
		}

		if ((strongNode = strongNode->parent))
		{
			checkChildren (strongNode);
		}
	}

	addToStrongBucket (strongRoot, &strongRoots[strongRoot->label]);
	++ highestStrongLabel;
}


static __inline void quickSort (Arc **arr, const uint first, const uint last)
{
/*************************************************************************
quickSort
*************************************************************************/
	int i=0, j, L=first, R=last, beg[MAX_LEVELS], end[MAX_LEVELS], temp=0;
	Arc *swap;

	if ((R-L) <= 5)
	{// Bubble sort if 5 elements or less
		for (i=R; (i>L); --i)
		{
			swap = NULL;
			for (j=L; j<i; ++j)
			{
				if (isExcess(arr[j]->flow - arr[j+1]->flow) < 0)//(arr[j]->flow < arr[j+1]->flow)
				{
					swap = arr[j];
					arr[j] = arr[j+1];
					arr[j+1] = swap;
				}
			}

			if (!swap)
			{
				return;
			}
		}

		return;
	}

	beg[0]=first;
	end[0]=last+1;

	while (i>=0)
	{
		L=beg[i];
		R=end[i]-1;

		if (L<R)
		{
			swap=arr[L];
			while (L<R)
			{
				while ((isExcess(arr[R]->flow - swap->flow)>=0) && (L<R)) //((arr[R]->flow >= swap->flow) && (L<R))
					R--;

				if (L<R)
				{
					arr[L]=arr[R];
					L++;
				}

				while ((isExcess(arr[L]->flow - swap->flow) <= 0) && (L<R)) //((arr[L]->flow <= swap->flow) && (L<R))
					L++;

				if (L<R)
				{
					arr[R]=arr[L];
					R--;
				}
			}

			arr[L] = swap;

			beg[i+1] = L+1;
			end[i+1] = end[i];
			end[i++] = L;

			if ((end[i]-beg[i]) > (end[i-1]-beg[i-1]))
			{
				temp=beg[i];
				beg[i]=beg[i-1];
				beg[i-1]=temp;
				temp=end[i];
				end[i]=end[i-1];
				end[i-1]=temp;
			}
		}
		else
		{
			i--;
		}
	}
}

static __inline void sort (Node * current)
{
/*************************************************************************
sort
*************************************************************************/
	if (current->numOutOfTree > 1)
	{
		quickSort (current->outOfTree, 0, (current->numOutOfTree-1));
	}
}

static __inline void minisort (Node *current)
{
/*************************************************************************
minisort
*************************************************************************/
	Arc *temp = current->outOfTree[current->nextArc];
	uint i, size = current->numOutOfTree;/*, tempflow = temp->flow;*/
	double tempflow = temp->flow;

	for(i=current->nextArc+1; ((i<size) && (isExcess(tempflow - current->outOfTree[i]->flow) < 0)); ++i)
	{
		current->outOfTree[i-1] = current->outOfTree[i];
	}
	current->outOfTree[i-1] = temp;
}


static /*ullint*/double checkOptimality (const uint gap)
{
/*************************************************************************
checkOptimality
*************************************************************************/
	uint i, check = 1;
	/*ullint*/double mincut = 0;
	/*llint*/ double *excess = NULL;

	Arc *tempArc;

	excess = (/*llint*/double *) mxMalloc(numNodes * sizeof (double/*llint*/));
	if (!excess)
	{
		printf("Out of memory\n");
		exit(0);
	}

	// Pushing depicits from all sink adjacent nodes to the sink
	for (i=0; i<nodesList[sink].numOutOfTree; ++i)
	{
		tempArc = nodesList[sink].outOfTree[i];
		if (isExcess(tempArc->from->excess) < 0)
		{
			if (isExcess((tempArc->from->excess + /*(int)*/ tempArc->flow))  < 0)
			{
				// Excess is high enough to saturate the arc => Flow on residual arc is zeroed
				tempArc->from->excess += /*(int)*/ tempArc->flow;
				tempArc->flow = 0;
			}
			else
				// Excess is NOT high enough to saturate the arc => Excess is zeroed
			{
				tempArc->flow = /*(uint)*/ (tempArc->from->excess + /*(int)*/ tempArc->flow);
				tempArc->from->excess = 0;
			}
		}
	}

	for (i=0; i<numNodes; ++i)
	{
		excess[i] = 0;
	}

	for (i=0; i<numArcs; ++i)
	{
		if ((arcList[i].from->label >= gap) && (arcList[i].to->label < gap))
		{
			mincut += arcList[i].capacity;
		}

		if ((isExcess(arcList[i].flow - arcList[i].capacity)>0) || (isExcess(arcList[i].flow) < 0))
		{
			check = 0;
			printf("Warning - Capacity constraint violated on arc (%d, %d). Flow = %d, capacity = %d\n",
				   arcList[i].from->number,
				   arcList[i].to->number,
				   arcList[i].flow,
				   arcList[i].capacity);
		}
		excess[arcList[i].from->number - 1] -= arcList[i].flow;
		excess[arcList[i].to->number - 1] += arcList[i].flow;
	}

	for (i=0; i<numNodes; i++)
	{
		if ((i != (source)) && (i != (sink)))
		{
			if (isExcess(excess[i]))
			{
				check = 0;
				printf ("Warning - Flow balance constraint violated in node %d. Excess = %lld\n",
						i+1,
						excess[i]);
			}
		}
	}

	check = 1;

	if (isExcess(excess[sink] - mincut) != 0)//(excess[sink] != mincut)
	{
		check = 0;
		printf("Warning - Flow is not optimal - max flow does not equal min cut!\n");
	}

// 	if (check)
// 	{
// 		printf ("Solution checks as optimal. \t Max Flow: \t %lld\n", mincut);
// 	}

	free (excess);
	excess = NULL;
	return mincut;
}

static void decompose (Node *excessNode, const uint source, uint *iteration)
{
/*************************************************************************
decompose
*************************************************************************/
	Node *current = excessNode;
	Arc *tempArc;
	/*uint*/double bottleneck = excessNode->excess;

	// Find the bottleneck along a path to the source or on a cycle
	for ( ;(current->number != source) && (current->visited < (*iteration)) /*&& (current->nextArc < current->numOutOfTree)*/; // Added by Cheng
		 current = tempArc->from)
	{
		current->visited = (*iteration);
		tempArc = current->outOfTree[current->nextArc];

		if (isExcess(tempArc->flow - bottleneck) < 0) //(tempArc->flow < bottleneck)
		{
			bottleneck = tempArc->flow;
		}
	}

	if (current->number == source) // the DFS reached the source
	{
		excessNode->excess -= bottleneck;
		current = excessNode;

		// Push the excess all the way to the source
		while (current->number != source)
		{
			tempArc = current->outOfTree[current->nextArc]; // Pick arc going out of node to push excess to
			tempArc->flow -= bottleneck; // Push back bottleneck excess on this arc

			if (isFlow(tempArc->flow)) // If there is still flow on this arc do sort on this current node
			{
				minisort(current);
			} else {
				++ current->nextArc;
			}

			current = tempArc->from;
		}
		return;
	}

	++ (*iteration);

	// This part is temporarily added by Cheng Lu.
//	if (current->visited == (*iteration)-1)
		bottleneck = current->outOfTree[current->nextArc]->flow;
/*	else{
		current = excessNode;
		bottleneck = excessNode->excess;
	}*/

	while (current->visited < (*iteration)) //&& (current->nextArc < current->numOutOfTree)) // Added by Cheng
	{
		current->visited = (*iteration);
		tempArc = current->outOfTree[current->nextArc];

		if (isExcess(tempArc->flow - bottleneck) < 0)//(tempArc->flow < bottleneck)
		{
			bottleneck = tempArc->flow;
		}
		current = tempArc->from;
	}

	++ (*iteration);

	// This part is temporarily added by Cheng Lu.
/*	if (current->visited == (*iteration)-1)
		;
	else{
		current = excessNode;
	}*/

	while (current->visited < (*iteration)) //&& (current->nextArc < current->numOutOfTree)) // Added by Cheng
	{
		current->visited = (*iteration);

		tempArc = current->outOfTree[current->nextArc];
		tempArc->flow -= bottleneck;

		if (isFlow(tempArc->flow))
		{
			minisort(current);
			current = tempArc->from;
		} else {
			++ current->nextArc;
			current = tempArc->from;
		}
	}
}

static void recoverFlow (const uint gap)
{
/*************************************************************************
recoverFlow
*************************************************************************/
	uint iteration = 1;
	uint i, j;
	Arc *tempArc;
	Node *tempNode;

	Node **nodePtrArray;

	if ((nodePtrArray = (Node **) mxMalloc(numNodes * sizeof (Node *))) == NULL)
	{
		printf("Out of memory\n");
		exit(0);
	}

	for (i=0; i < numNodes ; i++)
		nodePtrArray[i] = &nodesList[i];

	// Adding arcs FROM the Source to Source adjacent nodes.
	for (i=0; i<nodesList[source].numOutOfTree; ++i)
	{
		tempArc = nodesList[source].outOfTree[i];
		addOutOfTreeNode (tempArc->to, tempArc);
	}

	// Zeroing excess on source and sink nodes
	nodesList[source].excess = 0;
	nodesList[sink].excess = 0;

	for (i=0; i<numNodes; ++i)
	{
		tempNode = &nodesList[i];

		if ((i == (source)) || (i == (sink)))
		{
			continue;
		}

		if (tempNode->label >= gap) //tempNode is in SINK set
		{
			tempNode->nextArc = 0;
			if ((tempNode->parent) && (isFlow(tempNode->arcToParent->flow)))
			{
				addOutOfTreeNode (tempNode->arcToParent->to, tempNode->arcToParent);
			}

			for (j=0; j<tempNode->numOutOfTree; ++j)
			{ // go over all sink-set-node's arcs and look for arc with NO flow
				if (!isFlow(tempNode->outOfTree[j]->flow))
				{	// Remove arc with no flow
					-- tempNode->numOutOfTree;
					tempNode->outOfTree[j] = tempNode->outOfTree[tempNode->numOutOfTree];
					-- j;
				}
			}

			sort(tempNode);
		}
	}

	for (i=lowestPositiveExcessNode ; i < numNodes ; ++i)
	{
		tempNode = nodePtrArray[i];
		while (isExcess(tempNode->excess) > 0)
		{
			++ iteration;
			decompose(tempNode, source, &iteration);
		}
	}
	
	mxFree(nodePtrArray);
	nodePtrArray = NULL;
}

static void readData(const mxArray* arc_matrix_ptr, const mxArray* num_nodes_ptr, const mxArray* source_ptr, const mxArray* sink_ptr, const mxArray* lambda_range_ptr, const mxArray* round_ptr)
/*************************************************************************
readData
*************************************************************************/
{

	if (mxGetNumberOfDimensions(arc_matrix_ptr) > 2) {
		mexErrMsgTxt("Capacity matrix must be a 2D array");
	}
	else if (mxGetN(arc_matrix_ptr) != 4) {
		mexErrMsgTxt("Capacity matrix must a square matrix");
	}
	else if (mxIsEmpty(arc_matrix_ptr)) {
		mexErrMsgTxt("Input argument is empty\n");
	}
	else if (mxIsComplex(arc_matrix_ptr)) {
		mexErrMsgTxt("Capacity matrix must consists of real number (non-complex)");
	}
	else if (!mxIsNumeric(arc_matrix_ptr)) {
		mexErrMsgTxt("Capacity matrix must consists of numeric values");
	}

	numNodesSuper = (uint)mxGetScalar(num_nodes_ptr);
	numArcsSuper = mxGetM(arc_matrix_ptr);

	double* lambdaRange = mxGetPr(lambda_range_ptr);
	LAMBDA_LOW = lambdaRange[0];
	LAMBDA_HIGH = lambdaRange[1];
			
	roundNegativeCapacity = (uint)mxGetScalar(round_ptr);

	source = (uint)mxGetScalar(source_ptr) - 1;
	sink = (uint)mxGetScalar(sink_ptr) - 1;

	if ((source > numNodesSuper) || (source < 0) || (sink > numNodesSuper) || (sink < 0))
		mexErrMsgTxt("Error assigning source or sink nodes");

	double* arcMatrix = mxGetPr(arc_matrix_ptr);

	///* define parameters */
	uint arcCount = 0;
	uint numRemovedArcs = 0;
	uint from;
	uint to;
	double constantCapacity;
	double multiplierCapacity;

	uint i;

	//Initialize node arc datastructures
	if ((nodeListSuper = (Node *)mxMalloc(numNodesSuper * sizeof(Node))) == NULL)
	{
		printf("Could not allocate memory.\n");
		exit(0);
	}
	if ((arcListSuper = (Arc *)mxMalloc(numArcsSuper * sizeof(Arc))) == NULL)
	{
		printf("Could not allocate memory.\n");
		exit(0);
	}

	/* Initialization */
	for (i = 0; i < numNodesSuper; ++i)
	{
		initializeNode(&nodeListSuper[i], i);
		nodeListSuper[i].originalIndex = i;
	}
	for (i = 0; i < numArcsSuper; ++i)
	{
		initializeArc(&arcListSuper[i]);
	}

	if (LAMBDA_LOW == LAMBDA_HIGH)
	{
		useParametricCut = 0;
	}

	for (i = 0; i < numArcsSuper; ++i)
	{
		from = (uint)arcMatrix[ i ] - 1;
		to = (uint)arcMatrix[ 1 * numArcsSuper + i] - 1;
		constantCapacity = arcMatrix[ 2 * numArcsSuper + i];
		multiplierCapacity = arcMatrix[3 * numArcsSuper + i];

		/* assign arc */
		if (from < 0 || to < 0 || from >= numNodesSuper || to >= numNodesSuper)
		{
			printf("Nodes are labeled from 0 to <number of nodes>  - 1\n");
			exit(0);
		}
		else if (from == to)
		{
			printf("Node %u has a self loop which is not allowed\n", from);
			exit(0);
		}
		else if (multiplierCapacity > 0 && from != source)
		{
			printf("Only source adjacent arcs can have a strictly positive capacity multiplier\n");
			exit(0);
		}
		else if (multiplierCapacity < 0 && to != sink)
		{
			printf("Only sink adjacent arcs can have a strictly negative capacity multiplier\n");
			exit(0);
		}
		else if (to==source || from==sink)
		{
			numRemovedArcs++;
			continue;
		}

		arcListSuper[arcCount].constant = constantCapacity;
		arcListSuper[arcCount].multiplier = multiplierCapacity;
		arcListSuper[arcCount].from = &nodeListSuper[from];
		arcListSuper[arcCount].to = &nodeListSuper[to];

		++arcCount;

		++nodeListSuper[from].numAdjacent;
		++nodeListSuper[to].numAdjacent;
	}


	if (numRemovedArcs>0)
	{
	  Arc* arcListSuperAux;
	  numArcsSuper-= numRemovedArcs;
	  if ((arcListSuperAux = (Arc *)mxMalloc(numArcsSuper * sizeof(Arc))) == NULL)
	    {
	      printf("Could not allocate memory.\n");
	      exit(0);
	    }
	  for(i=0;i<numArcsSuper;++i){
	    arcListSuperAux[i].from = arcListSuper[i].from;
	    arcListSuperAux[i].to = arcListSuper[i].to;
	    arcListSuperAux[i].flow = arcListSuper[i].flow;
	    arcListSuperAux[i].capacity = arcListSuper[i].capacity;
	    arcListSuperAux[i].constant = arcListSuper[i].constant;
	    arcListSuperAux[i].direction = arcListSuper[i].direction;
	    arcListSuperAux[i].multiplier = arcListSuper[i].multiplier;
	  }
	  mxFree(arcListSuper);
	  arcListSuper = arcListSuperAux;
	}
	return;
}

static void pseudoflowPhase1 (void)
{
/*************************************************************************
pseudoflowPhase1
*************************************************************************/
	Node *strongRoot;
	while ((strongRoot = getHighestStrongRoot ()))
	{
		processRoot (strongRoot);
	}
}

static void removeDuplicateBreakpoints(void)
/*************************************************************************
removeDuplicateBreakpoints
*************************************************************************/
{
	Breakpoint *currentBreakpoint = firstBreakpoint;
	Breakpoint *nextBreakpoint = currentBreakpoint->next;

	while (nextBreakpoint != NULL)
	{
		if (currentBreakpoint->lambdaValue == nextBreakpoint->lambdaValue)
		{
			currentBreakpoint->next = nextBreakpoint->next;
			nextBreakpoint->next = NULL;
			destroyBreakpoint(nextBreakpoint);
		}
		currentBreakpoint = currentBreakpoint->next;
		// it may be that current breakpoint is null if there is one set of duplicate breakpoints
		if (currentBreakpoint != NULL)
		{
			nextBreakpoint = currentBreakpoint->next;
		}
		else
		{
			nextBreakpoint = NULL;
		}
	}
}

static void printOutput (mxArray* plhs[], double *times )
{
/*************************************************************************
printOutput
*************************************************************************/
	uint stats[5];
	Breakpoint *currentBreakpoint;
	uint numBreakpoints = 0;
	uint i;
	uint j;

	stats[0] = numArcScans;
	stats[1] = numMergers;
	stats[2] = numPushes;
	stats[3] = numRelabels;
	stats[4] = numGaps;

	/* count num breakpoints */
	currentBreakpoint = firstBreakpoint;
	while (currentBreakpoint != NULL)
	{
		++numBreakpoints;
		currentBreakpoint = currentBreakpoint->next;
	}

	plhs[0] = mxCreateDoubleMatrix(numNodesSuper, numBreakpoints, mxREAL);
	plhs[1] = mxCreateDoubleMatrix(1, numBreakpoints, mxREAL);
	plhs[2] = mxCreateDoubleMatrix(5, 1, mxREAL);
	plhs[3] = mxCreateDoubleMatrix(3, 1, mxREAL);
	
	double* cuts = mxGetPr(plhs[0]);
	double* lambdas = mxGetPr(plhs[1]);
	double* statsOut = mxGetPr(plhs[2]);
	double* timesOut = mxGetPr(plhs[3]);

	for (i = 0; i < 5; ++i) {
		statsOut[i] = (double)stats[i];
	}
		
	for (i = 0; i < 3; ++i) {
		timesOut[i] = (double)times[i];
	}
	
	
	/* print lambda values */
	currentBreakpoint = firstBreakpoint;
	for (i = 0; i < numBreakpoints; i++)
	{
		lambdas[i] = currentBreakpoint->lambdaValue;
		currentBreakpoint = currentBreakpoint->next;
	}

	/* print values nodes*/
	currentBreakpoint = firstBreakpoint;
	for (j = 0; j < numBreakpoints; j++)
	{
		for (i = 0; i < numNodesSuper; i++)
		{
		cuts[j * numNodesSuper + i ] = (double) currentBreakpoint->sourceSetIndicator[i];
		}
		currentBreakpoint = currentBreakpoint->next;
	}

	return;
}

static void copyArcNew(CutProblem *problem, int *nodeMap, Arc *old, Arc *new)
/*************************************************************************
copyArcNew - copy basic info arc and point to new nodes
*************************************************************************/
{
	uint newIndexFrom;
	uint newIndexTo;

	initializeArc(new);
	new->constant = old->constant;
	new->multiplier = old->multiplier;

	/* set start and end node */
	newIndexFrom = nodeMap[old->from->number];
	newIndexTo = nodeMap[old->to->number];
	new->from = &problem->nodeList[newIndexFrom];
	new->to = &problem->nodeList[newIndexTo];

	/* update degree nodes*/
	++ new->from->numAdjacent;
	++ new->to->numAdjacent;
}

static void copyArcAdd(Arc *old, Arc *new)
/*************************************************************************
copyArcAdd - update arc by adding another
*************************************************************************/
{
	new->constant += old->constant;
	new->multiplier += old->multiplier;
}

static void evaluateCapacities(CutProblem* problem)
/*************************************************************************
evaluateCapacities - Evaluate capacities for a particular value of capacities
*************************************************************************/
{
	uint i;
	for (i = 0; i < problem->numArcs; i++)
	{
		problem->arcList[i].capacity = problem->arcList[i].constant + problem->arcList[i].multiplier * problem->lambdaValue;
		if (problem->arcList[i].capacity < 0)
		{
			if (roundNegativeCapacity == 1 ||  problem->arcList[i].capacity > - TOL)
				problem->arcList[i].capacity = 0;
			else
			{
				printf("Negative capacity for lambda equal to %f. Set roundNegativeCapacity to 1 if the value should be rounded to 0.\n",problem->lambdaValue);
				exit(0);
			}
		}
	}
}

static void destroyProblem(CutProblem *problem)
/*************************************************************************
destroyProblem - Destruct function for CutProblem struct
*************************************************************************/
{
	mxFree(problem->sourceSet);
	problem->sourceSet = NULL;
	mxFree(problem->sinkSet);
	problem->sinkSet = NULL;
	mxFree(problem->nodeList);
	problem->nodeList = NULL;
	mxFree(problem->arcList);
	problem->arcList = NULL;
	mxFree(problem->optimalSourceSetIndicator);
	problem->optimalSourceSetIndicator = NULL;
}


static void initializeProblem(CutProblem * problem, Node *nodeListProblem, uint numNodesProblem, Arc *arcListProblem, uint numArcsProblem, const double lambdaValue)
/*************************************************************************
initializeProblem - Setup problems for parametric cut
*************************************************************************/
{
	uint i;
	uint currentNode = 2;
	int *nodeMap; /* indicator index of node in new nodeList. */

	/* set cut parameters */
	problem->cutValue = 0;
	problem->cutMultiplier = 0;
	problem->cutConstant = 0;

	/* set solved indicator */
	problem->solved = 0;

	/* initialize optimal cut */
	problem->optimalSourceSetIndicator = NULL;

	/* initialize new lambda value */
	problem->lambdaValue = lambdaValue;
	/* set size of node sets */
	problem->numSourceSet = 1;
	problem->numSinkSet = 1;
	problem->numNodesInList = numNodesProblem;

	/* allocateSpace for nodeMap */
	if ((nodeMap = (int *)mxMalloc(numNodesProblem* sizeof(int))) == NULL)
	{
		printf("Out of memory\n");
		exit(0);
	}

	/* allocate space for the node sets*/
	if ((problem->nodeList = (Node *)mxMalloc(problem->numNodesInList * sizeof(Node))) == NULL)
	{
		printf("Out of memory\n");
		exit(0);
	}
	if ((problem->sourceSet = (Node *)mxMalloc(problem->numSourceSet * sizeof(Node))) == NULL)
	{
		printf("Out of memory\n");
		exit(0);
	}
	if ((problem->sinkSet = (Node *)mxMalloc(problem->numSourceSet * sizeof(Node))) == NULL)
	{
		printf("Out of memory\n");
		exit(0);
	}

	/* create new node sets*/
	for (i = 0; i < numNodesProblem; i++)
	{
		if (i == source)
		{
			initializeNode( &problem->sourceSet[0], 0);
			problem->sourceSet[0].originalIndex = nodeListProblem[i].originalIndex;
			initializeNode(&problem->nodeList[0], 0); /* source is always first node */
			problem->nodeList[0].originalIndex = -1; /* indicate artificial source node */
			nodeMap[i] = 0;
		}
		else if (i == sink)
		{
			initializeNode(&problem->sinkSet[0], 0);
			problem->sinkSet[0].originalIndex = nodeListProblem[i].originalIndex;
			initializeNode(&problem->nodeList[1], 1); /* sink is always seond node */
			problem->nodeList[1].originalIndex = -2; /* indicate artificial sink node */
			nodeMap[i] = 1;
		}
		else
		{
			initializeNode(&problem->nodeList[currentNode], currentNode);
			problem->nodeList[currentNode].originalIndex = nodeListProblem[i].originalIndex;
			nodeMap[i] = currentNode;
			++currentNode;
		}
	}

	/* set number of arcs */
	problem->numArcs = numArcsProblem;

	/* allocate space for arcs */
	if ((problem->arcList = (Arc *)mxMalloc(problem->numArcs * sizeof(Arc))) == NULL)
	{
		printf("Out of memory\n");
		exit(0);
	}

	/* copy arcs */
	for (i = 0; i < numArcsProblem; i++)
	{
		copyArcNew(problem, nodeMap, &arcListProblem[i], &problem->arcList[i]);
	}

	/* evaluate capacities */
	evaluateCapacities(problem);

	/* free nodeMap*/
	mxFree(nodeMap);
	nodeMap = NULL;
}

static void contractProblem(CutProblem *problem, CutProblem *oldProblem, double lambdaValue, uint *lowSourceSetIndicator, uint *highSourceSetIndicator)
/*************************************************************************
contractProblem - create contracted instance based on lower bound and upper bound
*************************************************************************/
{
	uint i;
	uint originalIndex;
	uint currentNodeInList = 2;
	uint currentSourceSet = oldProblem->numSourceSet;
	uint currentSinkSet = oldProblem->numSinkSet;
	uint currentArc = 0;
	int *nodeMap; /* indicator index of node in new nodeList. */
	int *sourceAdjacentArcIndices;
	int *sinkAdjacentArcIndices;
	uint newIndexFrom, newIndexTo;

	/* set cut parameters */
	problem->cutValue = 0;
	problem->cutMultiplier = 0;
	problem->cutConstant = 0;

	/* set solved indicator */
	problem->solved = 0;

	/* initialize optimal cut */
	problem->optimalSourceSetIndicator = NULL;

	/* initialize new lambda value */
	problem->lambdaValue = lambdaValue;

	/* set size of node sets */
	problem->numSourceSet = oldProblem->numSourceSet;
	problem->numSinkSet = oldProblem->numSinkSet;
	problem->numNodesInList = oldProblem->numNodesInList;

	/* allocateSpace for nodeMap */
	if ((nodeMap = (int *)mxMalloc(oldProblem->numNodesInList* sizeof(int))) == NULL)
	{
		printf("Out of memory\n");
		exit(0);
	}

	/* determine size node sets - we can skip source and sink since they will stay*/
	for (i = 2; i < oldProblem->numNodesInList; i++)
	{
		originalIndex = oldProblem->nodeList[i].originalIndex;
		if (lowSourceSetIndicator[originalIndex] == 1)
		{
			++problem->numSourceSet;
			--problem->numNodesInList;
		}
		else if (highSourceSetIndicator[originalIndex] == 0)
		{
			++problem->numSinkSet;
			--problem->numNodesInList;
		}
	}

	/* allocate space for the node sets*/
	if ((problem->nodeList = (Node *)mxMalloc(problem->numNodesInList * sizeof(Node))) == NULL)
	{
		printf("Out of memory\n");
		exit(0);
	}
	if ((problem->sourceSet = (Node *)mxMalloc(problem->numSourceSet * sizeof(Node))) == NULL)
	{
		printf("Out of memory\n");
		exit(0);
	}
	if ((problem->sinkSet = (Node *)mxMalloc(problem->numSinkSet * sizeof(Node))) == NULL)
	{
		printf("Out of memory\n");
		exit(0);
	}

	/* copy source set */
	for (i = 0; i < oldProblem->numSourceSet; i++)
	{
		initializeNode(&problem->sourceSet[i], i);
		problem->sourceSet[i].originalIndex = oldProblem->sourceSet[i].originalIndex;
	}
	/* copy sink set */
	for (i = 0; i < oldProblem->numSinkSet; i++)
	{
		initializeNode(&problem->sinkSet[i], i);
		problem->sinkSet[i].originalIndex = oldProblem->sinkSet[i].originalIndex;
	}

	/* copy source */
	initializeNode(&problem->nodeList[0], 0); /* source is always first node */
	problem->nodeList[0].originalIndex = -1; /* indicate artificial source node */
	nodeMap[0] = 0;

	/* copy sink */
	initializeNode(&problem->nodeList[1], 1); /* sink is always seond node */
	problem->nodeList[1].originalIndex = -2; /* indicate artificial sink node */
	nodeMap[1] = 1;

	/* create new node list - we can ignore source and sink since they don't change */
	for (i = 2; i < oldProblem->numNodesInList; i++)
	{
		originalIndex = oldProblem->nodeList[i].originalIndex;
		if (lowSourceSetIndicator[originalIndex] == 1)
		{
			initializeNode(&problem->sourceSet[currentSourceSet], currentSourceSet);
			problem->sourceSet[currentSourceSet].originalIndex = oldProblem->nodeList[i].originalIndex;
			nodeMap[i] = 0;
			++currentSourceSet;
		}
		else if (highSourceSetIndicator[ originalIndex] == 0)
		{
			initializeNode(&problem->sinkSet[currentSinkSet], currentSinkSet);
			problem->sinkSet[currentSinkSet].originalIndex = oldProblem->nodeList[i].originalIndex;
			nodeMap[i] = 1;
			++currentSinkSet;
		}
		else
		{
			initializeNode(&problem->nodeList[currentNodeInList], currentNodeInList);
			problem->nodeList[currentNodeInList].originalIndex = oldProblem->nodeList[i].originalIndex;
			nodeMap[i] = currentNodeInList;
			++currentNodeInList;
		}
	}

	/* set number of arcs */
	problem->numArcs = oldProblem->numArcs;

	/* allocate space for source and sink arc indices */
	if ((sourceAdjacentArcIndices = (int *)mxMalloc(problem->numNodesInList *  sizeof(int))) == NULL)
	{
		printf("Out of memory\n");
		exit(0);
	}
	if ((sinkAdjacentArcIndices = (int *)mxMalloc(problem->numNodesInList *sizeof(int))) == NULL )
	{
		printf("Out of memory\n");
		exit(0);
	}

	/* initialize indices */
	for (i = 0; i < problem->numNodesInList; i++)
	{
		sourceAdjacentArcIndices[i] = -1;
		sinkAdjacentArcIndices[i] = -1;
	}

	/* determine new number of arcs */
	for (i = 0; i < oldProblem->numArcs; i++)
	{
		newIndexFrom = nodeMap[oldProblem->arcList[i].from->number];
		newIndexTo = nodeMap[oldProblem->arcList[i].to->number];
		if (newIndexFrom == newIndexTo || newIndexTo==0 || newIndexFrom==1)
		{
		}
		else if (newIndexFrom == 0)
		{
			if (sourceAdjacentArcIndices[newIndexTo] == -1)
			{
				sourceAdjacentArcIndices[newIndexTo] = currentArc;
				++currentArc;
			}
		}
		else if (newIndexTo == 1)
		{
			if (sinkAdjacentArcIndices[newIndexFrom] == -1)
			{
				sinkAdjacentArcIndices[newIndexFrom] = currentArc;
				++currentArc;
			}
		}
		else
		{
			++currentArc;
		}
	}

	/* set number of arcs */
	problem->numArcs = currentArc;

	/* allocate space for arcs */
	if ((problem->arcList = (Arc *)mxMalloc(problem->numArcs * sizeof(Arc))) == NULL)
	{
		printf("Out of memory\n");
		exit(0);
	}

	/* copy arcs */
	currentArc = 0;
	for (i = 0; i <oldProblem->numArcs; i++)
	{
		newIndexFrom = nodeMap[oldProblem->arcList[i].from->number];
		newIndexTo = nodeMap[oldProblem->arcList[i].to->number];
		if (newIndexFrom == newIndexTo || newIndexTo==0 || newIndexFrom==1)
		{
		}
		else if (newIndexFrom == 0)
		{
			if (sourceAdjacentArcIndices[newIndexTo] == currentArc )
			{
				copyArcNew(problem, nodeMap, &oldProblem->arcList[i], &problem->arcList[currentArc]);
				++currentArc;
			}
			else
			{
				copyArcAdd(&oldProblem->arcList[i], &problem->arcList[sourceAdjacentArcIndices[newIndexTo]]);
			}
		}
		else if (newIndexTo == 1)
		{
			if (sinkAdjacentArcIndices[newIndexFrom] == currentArc)
			{
				copyArcNew(problem, nodeMap, &oldProblem->arcList[i], &problem->arcList[currentArc]);
				++currentArc;
			}
			else
			{
				copyArcAdd(&oldProblem->arcList[i], &problem->arcList[sinkAdjacentArcIndices[newIndexFrom]]);
			}
		}
		else
		{
			copyArcNew(problem, nodeMap, &oldProblem->arcList[i], &problem->arcList[currentArc]);
			++currentArc;
		}
	}

	/* evaluate capacities */
	evaluateCapacities(problem );

	/* free local variables */
	mxFree(nodeMap);
	nodeMap = NULL;
	mxFree(sourceAdjacentArcIndices);
	sourceAdjacentArcIndices = NULL;
	mxFree(sinkAdjacentArcIndices);
	sinkAdjacentArcIndices = NULL;
}

static void initializeParametricCut(CutProblem *lowProblem, CutProblem *highProblem)
/*************************************************************************
initializeParametricCut - Set up data structures for parametric cut
*************************************************************************/
{
	/* initialize problem for LAMBDA_LOW */
	initializeProblem(lowProblem, nodeListSuper, numNodesSuper, arcListSuper, numArcsSuper, LAMBDA_LOW);


	if (useParametricCut == 1)
	{
		/* initialize problem for LAMBDA_HIGH */
		initializeProblem(highProblem, nodeListSuper, numNodesSuper, arcListSuper, numArcsSuper, LAMBDA_HIGH);
	}
}

static void addBreakpoint(double lambdaValue, uint *sourceSetIndicator)
/*************************************************************************
addBreakpoint - Adds a breakpoint to the linkedlist
*************************************************************************/
{
	Breakpoint *newBreakpoint;
	uint i;

	/* allocate memory for breakpoint*/
	if ((newBreakpoint= (Breakpoint*)mxMalloc(sizeof(Breakpoint))) == NULL)
	{
		printf("Could not allocate memory.\n");
		exit(0);
	}

	/* assign values */
	newBreakpoint->lambdaValue = lambdaValue;
	newBreakpoint->next = NULL;

	/* assign space for cut */
	if ((newBreakpoint->sourceSetIndicator = (uint*)mxMalloc(numNodesSuper * sizeof(uint))) == NULL)
	{
		printf("Could not allocate memory.\n");
		exit(0);
	}

	/* copy cut */
	for (i = 0; i < numNodesSuper; i++)
	{
		newBreakpoint->sourceSetIndicator[i] = sourceSetIndicator[i];
	}

	/* add breakpoint to linkedlist */
	if (lastBreakpoint == NULL)
	{
		/* initialize list */
		firstBreakpoint = newBreakpoint;
		lastBreakpoint = newBreakpoint;
	}
	else
	{
		/* add new element to linked list*/
		lastBreakpoint->next = newBreakpoint;
		/* update head */
		lastBreakpoint = newBreakpoint;
	}
}

static void createMemoryStructures( )
/*************************************************************************
createMemoryStructures - creates memory structures
*************************************************************************/
{
	uint from;
	uint to;
	uint i;
	double capacity;

	/* create memory structures */
	for (i=0; i<numNodes; ++i)
	{
		createOutOfTree(&nodesList[i]);
	}

	for (i=0; i<numArcs; i++)
	{
		to = arcList[i].to->number;
		from = arcList[i].from->number;
		capacity = arcList[i].capacity;

		if (!((source == to) || (sink == from) || (from == to)))
		{
			if ((source == from) && (to == sink))
			{
				arcList[i].flow = capacity;
			} else if (to == sink) {
				addOutOfTreeNode(&nodesList[to], &arcList[i]);
			} else {
				addOutOfTreeNode(&nodesList[from], &arcList[i]);
			}
		}
	}

	/* allocate memory for root and label count */
	if ((strongRoots = (Root *)mxMalloc(numNodes * sizeof(Root))) == NULL)
	{
		printf("Could not allocate memory.\n");
		exit(0);
	}
	if ((labelCount = (uint *)mxMalloc(numNodes * sizeof(uint))) == NULL)
	{
		printf("Could not allocate memory.\n");
		exit(0);
	}

	/* Initialization of root & labelcount */
	for (i = 0; i<numNodes; ++i)
	{
		initializeRoot(&strongRoots[i]);
		labelCount[i] = 0;
	}
}

static void evaluateCut(CutProblem *problem)
/*************************************************************************
evaluateCut - Evaluates optimal cut parameters for a given problem
*************************************************************************/
{
	uint i;
	int originalIndexFrom;
	int originalIndexTo;
	for (i = 0; i < problem->numArcs; ++i)
	{
		originalIndexFrom = problem->arcList[i].from->originalIndex;
		originalIndexTo = problem->arcList[i].to->originalIndex;
		if ((originalIndexFrom == -1 || problem->optimalSourceSetIndicator[originalIndexFrom] == 1) && (originalIndexTo == -2 || problem->optimalSourceSetIndicator[originalIndexTo] == 0 ) )
		{
		  problem->cutValue += problem->arcList[i].capacity;
		  problem->cutMultiplier += problem->arcList[i].multiplier;
		  problem->cutConstant += problem->arcList[i].constant;
		}
	}
}

static void solveProblem(CutProblem *problem, uint maximalSourceSet)
/*************************************************************************
solveProblem - solves a single instance of cut problem
*************************************************************************/
{
	uint i;
	uint *tempSourceSet;
	uint nodeCount;
	
	nodesList = problem->nodeList;
	numNodes = problem->numNodesInList;
	numArcs = problem->numArcs;
	problem->cutMultiplier = 0.0;
	problem->cutConstant = 0.0;
	problem->cutValue = 0.0;
	// handle empty problems
	if (numNodes == 2)
	{
		/* assign nodes to source / sink set */
		if ((problem->optimalSourceSetIndicator = (uint *)mxMalloc(numNodesSuper * sizeof(uint))) == NULL)
		{
			printf("Out of memory\n");
			exit(0);
		}

		for (i = 0; i < problem->numSourceSet; i++)
		{
			problem->optimalSourceSetIndicator[problem->sourceSet[i].originalIndex] = 1;
		}

		for (i = 0; i < problem->numSinkSet; i++)
		{
			problem->optimalSourceSetIndicator[problem->sinkSet[i].originalIndex] = 0;
		}

		/* determine cut value */
		for (i = 0; i < problem->numArcs; i++)
		{
			if (problem->arcList[i].from->originalIndex == -1 && problem->arcList[i].to->originalIndex == -2)
			{
				problem->cutConstant += problem->arcList[i].constant;
				problem->cutMultiplier += problem->arcList[i].multiplier;
				problem->cutValue += problem->arcList[i].capacity;
			}
		}

		return;
	}


	if (maximalSourceSet == 1)
	{
		source = 1;
		sink = 0;

		/* allocate space for reversed arcs */
		if ((arcList = (Arc *)mxMalloc(numArcs * sizeof(Arc))) == NULL)
		{
			printf("Out of memory\n");
			exit(0);
		}

		/* copy arcs such that arcs can be reversed */
		for (i = 0; i < numArcs; i++)
		{
			/* initialize new arc*/
			initializeArc(&arcList[i]);

			// reverse direction
			arcList[i].from = problem->arcList[i].to;
			arcList[i].to = problem->arcList[i].from;

			// assign capacity
			arcList[i].capacity = problem->arcList[i].capacity;
		}
	}
	else
	{
		source = 0;
		sink = 1;

		arcList = problem->arcList;
	}

	// solve
	createMemoryStructures();
	simpleInitialization();
	pseudoflowPhase1();

	/* allocate memory for source set (possibly reversed) */
	nodeCount = problem->numNodesInList + problem->numSourceSet + problem->numSinkSet - 2;
	if ((tempSourceSet = (uint *)mxMalloc(nodeCount * sizeof(uint))) == NULL)
	{
		printf("Out of memory\n");
		exit(0);
	}

	// retrieve optimal sourceSet for nodes in graph
	if (maximalSourceSet == 1) // reverse assignment to source and sink set
	{
		for (i = 2; i<numNodes; ++i) // start from 2 to ignore artificial source and sink
		{
			if (nodesList[i].label >= numNodes)
			{
				tempSourceSet[nodesList[i].originalIndex] = 0;
			}
			else
			{
				tempSourceSet[nodesList[i].originalIndex] = 1;
			}
		}
	}
	else
	{
		for (i = 2; i<numNodes; ++i) // start from 2 to ignore artificial source and sink
		{
			if (nodesList[i].label >= numNodes)
			{
				tempSourceSet[nodesList[i].originalIndex] = 1;
			}
			else
			{
				tempSourceSet[nodesList[i].originalIndex] = 0;
			}
		}
	}
	
	// process cut for source set nodes
	for (i = 0; i < problem->numSourceSet; i++)
	{
		tempSourceSet[problem->sourceSet[i].originalIndex] = 1;
	}
	// process cut for sink set nodes
	for (i = 0; i < problem->numSinkSet; i++)
	{
		tempSourceSet[problem->sinkSet[i].originalIndex] = 0;
	}

	// assign cut
	problem->optimalSourceSetIndicator = tempSourceSet;
	evaluateCut(problem);

	if (maximalSourceSet == 1)
	{
		// free if new memory has been allocated for arclist. Memory should not be freed if arcList is taken from the problem
		mxFree(arcList);
		arcList = NULL;
	}
	//printCutProblem(problem);
	freeMemorySolve();
}

static void parametricCut(CutProblem *lowProblem, CutProblem *highProblem)
/*************************************************************************
parametricCut - Recursive function that solves the parametric cut problem
*************************************************************************/
{
	
	double lambdaIntersect = 0;

	uint lambdaIntersectExists = 1;

	/* determine if this is the outermost recursion level*/
	int baseLevel = 0;
	if (lowProblem->solved == 0 && highProblem->solved == 0)
		baseLevel = 1;

	/* solve lower bound problem if necessary by finding minimal source set */
	if (lowProblem->solved == 0)
	{
		solveProblem(lowProblem, 0);
		lowProblem->solved = 1;
	}


	/* solve upper bound problem if nessecary by finding maximal source set */
	if (highProblem->solved == 0)
	{
		solveProblem(highProblem, 1);
		highProblem->solved = 1;
	}

	/* find lambda value for which the optimal cut functions(expressed as a function of lambda) for the lower bound and upper bound problem intersect. */
	if (dabs(highProblem->cutMultiplier - lowProblem->cutMultiplier) > TOL)
	{
		lambdaIntersect = (lowProblem->cutConstant - highProblem->cutConstant) / (highProblem->cutMultiplier - lowProblem->cutMultiplier);
		lambdaIntersectExists = 1;
	}
	else // conclude that there is no intersection if denominator is too close to zero
	{
		lambdaIntersectExists = 0;
	}

	/* check cases depending on intersection value of the lower bound and upper bound optimal cut. */
	if (lambdaIntersectExists == 1 && lambdaIntersect + TOL < highProblem->lambdaValue && lambdaIntersect - TOL > lowProblem->lambdaValue)
	{
		/* if intersection occurs strictly within the interval, then there are at least 2 breakpoints.lambdaIntersect is guaranteed to separate the interval into subintervals each containing at least 1 breakpoint. */

		/* Create new instance of upper bound problem with contracted optimal source set from the low problem and the sink set from the optimal cut for the high problem and lambda value equal to lambda intersect. The nodes in the source set for lambdaLow are guaranteed to be in the source set for the lambda >= lambdaLow. The nodes that are in the sinkset for lambdaHigh are guaranteed to be in the sink set for lambda <= lambdaIntersect <= lambdaHigh. */
		CutProblem upperBoundIntersect;
		contractProblem(&upperBoundIntersect, lowProblem, lambdaIntersect, lowProblem->optimalSourceSetIndicator, highProblem->optimalSourceSetIndicator);

		/* recurse for lower subinterval */
		parametricCut(lowProblem, &upperBoundIntersect);

		/* Create new instance of upper bound problem with contracted optimal source set from the low problem and the sink set from the optimal cut for the high problem and lambda value equal to lambda intersect. The nodes in the source set for lambdaLow are guaranteed to be in the source set for the lambda >= lambdaLow. The nodes that are in the sinkset for lambdaHigh are guaranteed to be in the sink set for lambda <= lambdaIntersect <= lambdaHigh. */
		CutProblem lowerBoundIntersect;
		contractProblem(&lowerBoundIntersect, lowProblem, lambdaIntersect,lowProblem->optimalSourceSetIndicator, highProblem->optimalSourceSetIndicator);

		/* recurse for higher subinterval */
		parametricCut(&lowerBoundIntersect, highProblem);

		/* call destructor function */
		destroyProblem(&lowerBoundIntersect);
		destroyProblem(&upperBoundIntersect);
	}
	else if (lambdaIntersectExists == 1 && dabs( lambdaIntersect - highProblem->lambdaValue ) <= TOL )
	{
		/* if lambda intersect is equal to upper bound, then lambdaHigh is a breakpoint and no further recursion necessary. */
		addBreakpoint(highProblem->lambdaValue, lowProblem->optimalSourceSetIndicator);
	}
	else if (lambdaIntersectExists == 1 && dabs( lambdaIntersect - lowProblem->lambdaValue ) <= TOL )
	{
		/* if lambda intersect is equal to lower bound, then lambdaLow is a breakpoint and no further recursion necessary.*/
		addBreakpoint(lowProblem->lambdaValue, lowProblem->optimalSourceSetIndicator);
	}

	/* add cut corresponding to lambdaHigh iff first recursion level */
	if (baseLevel == 1)
	{
		addBreakpoint(highProblem->lambdaValue, highProblem->optimalSourceSetIndicator);
	}
}

void resetGlobal(void)
/*************************************************************************
resetGlobal - reset global variables
*************************************************************************/
{
	TOL = 1E-8;
	numNodes = 0;
	numArcs = 0;
	numNodesSuper = 0;
	numArcsSuper = 0;
	source = 0;
	sink = 0;
	highestStrongLabel = 1;

	numArcScans = 0;
	numPushes = 0;
	numMergers = 0;
	numRelabels = 0;
	numGaps = 0;

	nodesList = NULL;
	strongRoots = NULL;
	labelCount = NULL;
	arcList = NULL;
	nodeListSuper = NULL;
	arcListSuper = NULL;
	lowestPositiveExcessNode = 0;

	lastBreakpoint = NULL;
	firstBreakpoint = NULL;

	useParametricCut = 1;
	roundNegativeCapacity = 0;

	LAMBDA_LOW = 0;
	LAMBDA_HIGH = 0;

	return;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray*prhs[])
/*************************************************************************
mexFunction - Main function
*************************************************************************/
{
	resetGlobal();
	
	/*ullint*/double flow = 0;
	double readStart, readEnd, initStart, initEnd, solveStart, solveEnd;

	numArcScans = 0;
	numMergers = 0;
	numPushes = 0;
	numRelabels = 0;
	numGaps = 0;

	/* Check for proper arguments */
	if (nrhs != 6) {
		mexErrMsgTxt("Usage [cuts, lambdas, stats, times] = hpf(arc matrix, num_nodes, source node, sink node, lambda range, rounding)");
	}
	else if (nlhs > 4) {
		mexErrMsgTxt("Too many output arguments.");
	}
	else if (nlhs < 4) {
		mexErrMsgTxt("Too few output arguments.");
	}


	readStart = clock();
	// readInput
	readData( ARC_MATRIX, NUM_NODES, SOURCE, SINK, LAMBDA_RANGE, ROUNDING );
	readEnd = clock();

	initStart = clock();
	CutProblem lowProblem;
	CutProblem highProblem;
	initializeParametricCut(&lowProblem, &highProblem);
	initEnd = clock();

	solveStart = clock();
	if (useParametricCut == 1)
	{
		parametricCut(&lowProblem, &highProblem);
		/* deallocate memory */
		removeDuplicateBreakpoints();
		destroyProblem(&lowProblem);
		destroyProblem(&highProblem);
	}
	else
	{
		solveProblem(&lowProblem, 0);
		/* add solution as breakpoint */
		addBreakpoint(lowProblem.lambdaValue, lowProblem.optimalSourceSetIndicator);
		/* deallocate memory */
		destroyProblem(&lowProblem);
	}
	solveEnd = clock();

	double times[3];
	times[0] = (readEnd - readStart) / CLOCKS_PER_SEC;
	times[1] = (initEnd - initStart) / CLOCKS_PER_SEC;
	times[2] = (solveEnd - solveStart) / CLOCKS_PER_SEC;

	///* RECOVER FLOW NEEDS TO BE ADAPTED TO DEAL WITH PARAMETRIC ALGORITHM */
	////	recoverFlow( numNodes );
	////	flow = checkOptimality (numNodes);

	printOutput(plhs, times);

	freeMemoryComplete();

	return;

}