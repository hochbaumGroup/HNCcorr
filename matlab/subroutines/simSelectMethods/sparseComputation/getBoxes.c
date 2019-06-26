#include "math.h"
#include "mex.h"

/*************************************************************************
Definitions
*************************************************************************/
typedef struct Box {
	int id;
	int counter;
	int numNeighbors;
	int numFeatureVectors;
	int *featureVectors;
	struct Box *next;
	int *neighbors;
} Box;

/*************************************************************************
Global variables
*************************************************************************/
static double numVec;
static double numDim;
static double gridRes;

static Box **boxList = NULL;

static void initializeBox (Box *bx,int id)
{
//*************************************************************************
//initializeBox
//*************************************************************************/
	bx->id = id;
	bx->numNeighbors = 0;
	bx->numFeatureVectors = 0;
	bx->featureVectors = NULL;
	bx->neighbors = NULL;
	bx->next = NULL;
}

static void freeBox (Box *bx)
{
//*************************************************************************
//freeBox
//*************************************************************************/
	mxFree(bx->featureVectors);
	if(bx->numNeighbors>0) {
		mxFree(bx->neighbors);
	}
	mxFree(bx);
}

double** computeJumps (double **scores, double *min_score, double *max_score)
{
//*************************************************************************
//computeJumps
//*************************************************************************/
	double *jumps;
	int i,j;

	jumps = (double *) mxMalloc(sizeof(double)*numDim);

	// Get min and max values for each dimension of the scores
	for (i=0;i<numDim;i++) {
		jumps[i] = (max_score[i]-min_score[i])/(gridRes);
		//for (j = 0; j<numVec; j++) {
		//	scores[j][i] = scores[j][i]-min;
		//}
	}


	return jumps;
}

int getID(int *labels) {
/*************************************************************************
getID
*************************************************************************/
	int i,id;

	id = 0;
	for (i=0;i<numDim;i++) {
		id = id + (labels[i]-1)*pow(gridRes,numDim-(i+1));
	}
	id = id + 1;
	return id;
}

int* getLabels (double *scores,double *jumps, double* min_score,double* max_score)
{
//*************************************************************************
//getLabels
//*************************************************************************/
	int *labels;
	int i,j,id;

	labels = (int *) mxMalloc(sizeof(int)*numDim);

	for (i = 0; i < numDim; i++) {
		if (scores[i] == max_score[i] ) {
			labels[i] = floor( (scores[i] - min_score[i] ) / jumps[i]);
		}
		else {
			labels[i] = floor((scores[i] - min_score[i]) / jumps[i]) + 1;
		}

	}

	return labels;
}

int getNeighbors (int *labels, Box *bx)
{
//*************************************************************************
//getNeighbors
//*************************************************************************/
	int i,j;
	int maxNeighbors,numNeighbors;
	int *neighbor_indices;
	int *val,*divisor,*counter,*tmpLabels;

	val = (int *)mxMalloc(sizeof(int)*numDim);
	divisor = (int *)mxMalloc(sizeof(int)*numDim);
	counter = (int *)mxMalloc(sizeof(int)*numDim);
	tmpLabels = (int *)mxMalloc(sizeof(int)*numDim);

	maxNeighbors = pow(3,numDim);
	numNeighbors = 0;

	neighbor_indices = (int *)mxMalloc(sizeof(int)*maxNeighbors);

	// Copy label values
	for (i=0;i<numDim;i++) {
		val[i] = labels[i]-1;
		divisor[i] = pow(3,numDim-(i+1));
		counter[i] = 0;
	}

	// Get indices of neighboring boxes
	for(i=0;i<maxNeighbors;i++){
		for(j=0;j<numDim;j++) {
			tmpLabels[j] = val[j];

			if ((i+1) % divisor[j] == 0) {
				val[j]++;
				counter[j]++;
				if (counter[j] > 2) {
					val[j] -= 3;
					counter[j] -=3;
				}
			}
		}
		if (isFeasible(labels,tmpLabels)) {
			neighbor_indices[numNeighbors] = getID(tmpLabels);
			numNeighbors++;
		}
	}

	// Assign pointers
	bx->neighbors = (int *) mxMalloc(sizeof(Box *)*numNeighbors);
	for (i=0;i<numNeighbors;i++) {
		bx->neighbors[i] = neighbor_indices[i];
	}

	mxFree(val);
	mxFree(divisor);
	mxFree(counter);
	mxFree(tmpLabels);
	mxFree(neighbor_indices);

	return numNeighbors;
}

int isFeasible(int *labels, int *tmpLabels) {
//*************************************************************************
//isFeasible
//*************************************************************************/
	int i,feasible,different;

	feasible = 1;
	different = 0;

	for (i=0;i<numDim;i++) {
		if (tmpLabels[i] < 1 || tmpLabels[i] > gridRes) {
			feasible = 0;
			break;
		}
		if (tmpLabels[i] != labels[i]) {
			different = 1;
		}
	}

	return feasible && different;
}

void compressListOfNeighbors(Box *bx) {
//*************************************************************************
//compressListOfNeighbors
//*************************************************************************/
	int i,counter;
	int *neighbor_indices;

	neighbor_indices = (int *) mxMalloc(sizeof(int)*bx->numNeighbors);
	counter = 0;
	for(i=0;i<bx->numNeighbors;i++) {
		if (boxList[bx->neighbors[i]-1] != NULL && bx->neighbors[i] > bx->id) {
				neighbor_indices[counter] = bx->neighbors[i];
				counter++;
		}
	}

	bx->numNeighbors = counter;
	bx->neighbors = (int *) mxRealloc(bx->neighbors,sizeof(int)*counter);
	for (i=0;i<bx->numNeighbors;i++) {
		bx->neighbors[i] = neighbor_indices[i];
	}

	mxFree(neighbor_indices);
}

Box* createGrid(double **scores, double *min_score, double *max_score) {
//*************************************************************************
//createGrid
//*************************************************************************/
	int i,numBoxes,counter,rep;
	int *labels;
	Box *start,*currentBox;
	double *jumps;

	numBoxes = (int) pow(gridRes,numDim);
	boxList = (Box **) mxMalloc(sizeof(Box*)*numBoxes);

	// Analyze scores
	jumps = computeJumps(scores, min_score, max_score);

	// Initialize boxes (temporary for debugging purposes)
	for(i=0;i<numBoxes;i++) {
		boxList[i] = NULL;
	}

	// Create boxes
	for (i=0;i<numVec;i++) {
		labels = getLabels(scores[i],jumps,min_score,max_score);
		rep = getID(labels);
		if (boxList[rep-1] == NULL) {
			boxList[rep-1] = (Box *) mxMalloc(sizeof(Box));
			initializeBox(boxList[rep-1],rep);
			boxList[rep-1]->numNeighbors = getNeighbors(labels,boxList[rep-1]);
		}
		boxList[rep-1]->numFeatureVectors++;
		mxFree(labels);
	}
	// Allocate memory for feature vectors
	for (i=0;i<numBoxes;i++) {
		if (boxList[i] != NULL) {
			boxList[i]->featureVectors = (int *) mxMalloc(sizeof(int)*(boxList[i]->numFeatureVectors));
			boxList[i]->counter = 0;
		}
	}

	// Assign feature vectors to representatives
	for (i=0;i<numVec;i++) {
		labels = getLabels(scores[i],jumps,min_score,max_score);
		rep = getID(labels);
		boxList[rep-1]->featureVectors[boxList[rep-1]->counter] = i+1;
		boxList[rep-1]->counter++;
		mxFree(labels);
	}

	// Compress list

	// Get first non-empty box
	for (i=0;i<numBoxes;i++) {
		if (boxList[i] != NULL) {
			start = boxList[i];
			break;
		}
	}

	// Compress list
	if (start != NULL) {
		currentBox = start;
		counter = i+1;
		while (counter<numBoxes) {
			if (boxList[counter] != NULL) {
				currentBox->next = boxList[counter];
				currentBox = boxList[counter];
			}
			counter++;
		}
	}

	// Compress list of neighbors
	if (start != NULL) {
		currentBox = start;
		while (currentBox != NULL) {
			compressListOfNeighbors(currentBox);
			currentBox = currentBox->next;
		}
	}

	mxFree(jumps);

	return start;
}

void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
	double *scores_list;
	double **scores;
	int numBoxes,numOfPairsOfNeighboringBoxes,numNonEmptyBoxes;
	mxArray *o;
	int i,j,counter,boxCounter;
	double *min_score, *max_score;
	double *vals;
	mwSize ndim;
	mwSize *dims;
	Box *startOfList,*currentBox,*previousBox;

	/* Check for proper number of input and output arguments */
	if (nrhs != 4) {
		mexErrMsgIdAndTxt( "MATLAB:invalidNumInputs",
				"Four input arguments required.");
	}
	if(nlhs > 3){
		mexErrMsgIdAndTxt( "MATLAB:maxlhs",
				"Too many output arguments.");
	}

	/* Check data type of input arguments  */
	if (!(mxIsDouble(prhs[0])) || !(mxIsDouble(prhs[1]))){
		mexErrMsgIdAndTxt( "MATLAB:inputNotDouble",
				"Input arguments must be of type double");
	}

	// Get scores
	scores_list = mxGetPr(prhs[0]);
	numVec =  mxGetM(prhs[0]);
	numDim = mxGetN(prhs[0]);

	// Get grid resolution
	gridRes = mxGetScalar(prhs[1]);
	
	// Get min and max scores
	min_score = mxGetPr(prhs[2]);
	max_score = mxGetPr(prhs[3]);

	scores = (double **)mxMalloc(sizeof(double*)*numVec);
	// Transfer list to matrix
	for (i=0;i<numVec;i++) {
		scores[i] = (double *)mxMalloc(sizeof(double)*numDim);
		for (j=0;j<numDim;j++) {
			scores[i][j] = scores_list[(int) numVec*j + i];
		}
	}

	// Set parameters
	numBoxes = (int) pow(gridRes,numDim);

	// Create grid
	startOfList = createGrid(scores, min_score, max_score);

	// Get number of non-empty boxes
	numNonEmptyBoxes = 0;
	boxCounter = 0;
	currentBox = startOfList;
	while(currentBox != NULL) {
		if (currentBox->numFeatureVectors > 0)
			boxCounter++;
		currentBox = currentBox->next;
		numNonEmptyBoxes++;
	}

	// Get number of pairs of neighboring boxes
	numOfPairsOfNeighboringBoxes = 0;
	currentBox = startOfList;
	while(currentBox != NULL) {
		numOfPairsOfNeighboringBoxes = numOfPairsOfNeighboringBoxes + currentBox->numNeighbors;
		currentBox = currentBox->next;
	}

	// Initialize output arrays
	ndim = 2;
	dims = (mwSize *) mxMalloc(ndim*sizeof(mwSize));
	dims[0] = boxCounter;
	dims[1] = 1;
    plhs[0] = mxCreateCellArray(ndim,dims);

	dims[0] = numOfPairsOfNeighboringBoxes;
	dims[1] = 2;
    plhs[1] = mxCreateCellArray(ndim,dims);

	dims[0] = numOfPairsOfNeighboringBoxes;
	dims[1] = 2;
	plhs[2] = mxCreateCellArray(ndim, dims);

	// Set value of output array boxes
	currentBox = startOfList;
	counter = 0;
	while(currentBox != NULL) {
		if (currentBox->numFeatureVectors > 0) {
			o = mxCreateDoubleMatrix(1, currentBox->numFeatureVectors,mxREAL);
			vals = mxGetPr(o);

			for (j=0;j<currentBox->numFeatureVectors;j++) {
				vals[j] = currentBox->featureVectors[j];
			}

			mxSetCell(plhs[0], counter, o);
			counter++;
		}
		currentBox = currentBox->next;
	}

	// Set value of output array neighbors
	if (numOfPairsOfNeighboringBoxes > 0) {
		currentBox = startOfList;
		counter = 0;
		while(currentBox != NULL) {

			for (i=0;i<currentBox->numNeighbors;i++) {
				o = mxCreateDoubleMatrix(1, currentBox->numFeatureVectors,mxREAL);
				vals = mxGetPr(o);

				for (j=0;j<currentBox->numFeatureVectors;j++) {
					vals[j] = currentBox->featureVectors[j];
				}

				mxSetCell(plhs[1], counter, o);

				o = mxCreateDoubleMatrix(1, boxList[currentBox->neighbors[i]-1]->numFeatureVectors,mxREAL);
				vals = mxGetPr(o);

				for (j=0;j<boxList[currentBox->neighbors[i]-1]->numFeatureVectors;j++) {
					vals[j] = boxList[currentBox->neighbors[i]-1]->featureVectors[j];
				}
				mxSetCell(plhs[1], numOfPairsOfNeighboringBoxes + counter, o);
				counter++;
			}
			currentBox = currentBox->next;
		}
	}

	// Relabel box ids
	boxCounter = 1;
	currentBox = startOfList;
	while (currentBox != NULL) {
		currentBox->id = boxCounter++;
		currentBox = currentBox->next;
	}


	// Set value of output array box mapping
	if (numOfPairsOfNeighboringBoxes > 0) {
		currentBox = startOfList;
		counter = 0;
		while (currentBox != NULL) {

			for (i = 0; i<currentBox->numNeighbors; i++) {
				o = mxCreateDoubleMatrix(1, 1, mxREAL);
				vals = mxGetPr(o);
				vals[0] = currentBox->id;

				mxSetCell(plhs[2], counter, o);

				o = mxCreateDoubleMatrix(1, 1, mxREAL);
				vals = mxGetPr(o);
				vals[0] = boxList[currentBox->neighbors[i] - 1]->id;

				mxSetCell(plhs[2], numOfPairsOfNeighboringBoxes + counter, o);
				counter++;
			}
			currentBox = currentBox->next;
		}
	}

	mxFree(dims);
	// Free boxes
	currentBox = startOfList;
	while (currentBox != NULL) {
		previousBox = currentBox;
		currentBox = currentBox->next;
		freeBox(previousBox);
	}
	mxFree(boxList);
	for(i=0;i<numVec;i++)
		mxFree(scores[i]);
	mxFree(scores);

    return;
}


