#include "kmeans.h"

/* Helper Functions */

//called by assignClusters
__device__ float computeDistance(Point* point1, Point* point2, int dims) {
    float dist;
        
    for(int i = 0; i < dims; i++) {
        if(point1->coords[i] != NULL && point2->coords[i]) {
            float temp = fabs(point1->coords[i] - point2->coords[i]);
            temp = temp * temp;
            dist += temp;
        }
    }
    return (float) sqrt(dist);
}

/* FIRST CUDA IMPLEMENTATION */

__global__ void assignClusters(Point *devicePointArr, Point *deviceCentroidArr, int nPoints, int num_clusters, int dims) {
    
    //id of thread - represents one point in devicePointArr
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index < nPoints) {
        if(&devicePointArr[index] == NULL) {
            printf("Null pointer\n");
            return;
        }

        for(int i = 0;  i < num_clusters; i++) {
            if(&deviceCentroidArr[i] == NULL) {
                printf("Null pointer\n");
                return;
            }
            float dist = computeDistance(&devicePointArr[index], &deviceCentroidArr[i], dims);
            if(dist < devicePointArr[index].minDistance) {
                atomicExch(&devicePointArr[index].minDistance, dist);
                atomicExch(&devicePointArr[index].clusterID, i);
            }
        }
    }
}

//computes sums through reduction
__global__ void sumDimensions(Point *devicePointArr, int *deviceClusterSize, float* deviceSumDimensions, size_t pitch,
        int nPoints, int num_clusters, int dims) {
    
    //printf("Summing Dimensions");

    //id of thread - should represent a point in devicePointArr
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index >= nPoints) {
        return;
    }
    
    //sum dimensions of given point
    atomicAdd(&deviceClusterSize[devicePointArr[index].clusterID], 1);

    float* row = (float*)((char*)deviceSumDimensions + devicePointArr[index].clusterID * pitch);
    for(int j = 0; j < dims; j++) {
        atomicAdd(&row[j], devicePointArr[index].coords[j]);
    }
    atomicExch(&devicePointArr[index].minDistance, __FLT_MAX__);
}

//recomputes centroids based on mean of clusters
__global__ void updateCentroids(Point *deviceCentroidArr , int *deviceClusterSize, float* deviceSumDimensions, size_t pitch,
        int nPoints, int num_clusters, int dims, bool* updated, double threshold) {
    
    //id of thread - should represent a centroid in deviceCentroidArr
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index >= num_clusters) {
        return;
    }

    if(deviceClusterSize[index] != 0) {
        for(int j = 0; j < dims; j++) {
            float* row = (float*)((char*) deviceSumDimensions + index * pitch);
            float meanVal = row[j] / float(deviceClusterSize[index]);

            //check if centroids have changed by threshold amount, if not then we've converged
            if(fabs(deviceCentroidArr[index].coords[j] - meanVal) > float(threshold)) {
                updated[0] = true;
            }
            atomicExch(&deviceCentroidArr[index].coords[j], meanVal);
        }
    }
}

/* CUDA WITH SHARED MEMORY IMPLEMENTATION */

//recomputes centroids based on mean of clusters
__global__ void updateCentroidsSharedMem(Point *devicePointArr, Point *deviceCentroidArr, int *deviceClusterSize, float* deviceSumDimensions,
        size_t pitch, int nPoints, int num_clusters, int dims, bool* updated, double threshold) {
    
    const int index = blockIdx.x * blockDim.x + threadIdx.x;  //index out of all threads (i.e. nPoints)
    const int threadIndex = threadIdx.x;  //index of thread within its block (threads per block = 1024)

    if(index >= nPoints) {
        return;
    }

    //do this here instead of when summing dimensions
    atomicExch(&devicePointArr[index].minDistance, __FLT_MAX__);

    //setup shared memory and store given index from pointArr
    extern __shared__ Point sharedPointArr[];
	sharedPointArr[threadIndex] = devicePointArr[index];

    __syncthreads();

    //update shared memory - first thread in each block computes the sums for its entire block(rest solely input and copy back data)
    if(threadIndex == 0) {
        //sum values for block and atomically add to device arrays
        for(int i = 0; i < blockDim.x; i++) {
            atomicAdd(&deviceClusterSize[sharedPointArr[i].clusterID], 1);

            float* row = (float*)((char*) deviceSumDimensions + sharedPointArr[i].clusterID * pitch);
            for(int j = 0; j < dims; j++) {
                atomicAdd(&row[j], sharedPointArr[i].coords[j]);
            }
            //atomicExch(&devicePointArr[i].minDistance, __FLT_MAX__);
        }
    }

    __syncthreads();

    //have the first num_clusters threads pass results back to gmem (only need one thread per centroid)
    if(index < num_clusters) {
        if(deviceClusterSize[index] != 0) {
            for(int j = 0; j < dims; j++) {
                float* row = (float*)((char*) deviceSumDimensions + index * pitch);
                float meanVal = row[j] / float(deviceClusterSize[index]);

                if(fabs(deviceCentroidArr[index].coords[j] - meanVal) > float(threshold)) {
                    updated[0] = true;
                }
                
                //update centroid with new coords
                atomicExch(&deviceCentroidArr[index].coords[j], meanVal);
            }
        }
    }
}

/* K-Means++ IMPLEMENTATION */
__global__ void computeDValue(Point *devicePointArr, Point *deviceCentroidArr, float* deviceDValues, int nPoints, int num_clusters, int dims) {
    //id of thread - represents one point in devicePointArr
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(index < nPoints) {
        for(int i = 0;  i < num_clusters; i++) {
            float dist = computeDistance(&devicePointArr[index], &deviceCentroidArr[i], dims);
            if(dist < deviceDValues[index]) {
                atomicExch(&deviceDValues[index], dist);
            }
        }
    }
}

/* KERNEL WRAPPERS */
void assignClustersKernel(Point *devicePointArr, Point *deviceCentroidArr, int nPoints, int num_clusters, int dims) {
    int numThreads = 1024;
    int numBlocks = (nPoints + numThreads - 1) / numThreads;
    assignClusters<<<numBlocks, numThreads>>>(devicePointArr, deviceCentroidArr, nPoints, num_clusters, dims);
}

void sumDimensionsKernel(Point *devicePointArr, int *deviceClusterSize, float* deviceSumDimensions, size_t pitch,
        int nPoints, int num_clusters, int dims) {
    int numThreads = 1024;
    int numBlocks = (nPoints + numThreads - 1) / numThreads;
    sumDimensions<<<numBlocks, numThreads>>>(devicePointArr, deviceClusterSize, deviceSumDimensions, pitch,
                    nPoints, num_clusters, dims);
}

void updateCentroidsKernel(Point *deviceCentroidArr , int *deviceClusterSize, float* deviceSumDimensions, size_t pitch,
        int nPoints, int num_clusters, int dims, bool* updated, double threshold) {
    updateCentroids<<<1, num_clusters>>>(deviceCentroidArr, deviceClusterSize, deviceSumDimensions, pitch, nPoints, num_clusters, dims, 
                        updated, threshold);
}

void updateCentroidsSMKernel(Point *devicePointArr, Point *deviceCentroidArr, int *deviceClusterSize, float* deviceSumDimensions, size_t pitch,
        int nPoints, int num_clusters, int dims, bool* updated, double threshold) {
    int numThreads = 1024;
    int numBlocks = (nPoints + numThreads - 1) / numThreads;
    updateCentroidsSharedMem<<<numBlocks, numThreads, numThreads * sizeof(Point)>>>(devicePointArr, deviceCentroidArr, deviceClusterSize, 
                                deviceSumDimensions, pitch, nPoints, num_clusters, dims, updated, threshold);
}

void computeDValueKernel(Point *devicePointArr, Point *deviceCentroidArr, float* deviceDValues, int nPoints, int num_clusters, int dims) {
    int numThreads = 1024;
    int numBlocks = (nPoints + numThreads - 1) / numThreads;
    computeDValue<<<numBlocks, numThreads>>>(devicePointArr, deviceCentroidArr, deviceDValues, nPoints, num_clusters, dims);
}