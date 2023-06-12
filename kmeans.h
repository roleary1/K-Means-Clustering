#ifndef KMEANS_H
#define KMEANS_H

#include <vector>
#include <chrono> 
#include <ctime>
#include <tuple>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <exception>

#include <cuda.h>
#include <cuda_runtime_api.h>

struct Point;   //forward declaration

/* Command Line Arguments */
extern int num_clusters, dims, max_num_iter, seed;
extern bool cFlag, gFlag, fFlag, pFlag;
extern double threshold;
extern std::string inputFileName, answerFileName;

/* Global Variables for Computation */
extern int nPoints;        //total number of points
extern Point* pointArr;       
extern std::vector<Point> centroidArr;

extern bool* updated;

/* Timing Variables */
extern int iter_to_converge;
extern float time_per_iter_in_ms;

struct Point {
    float* coords;
    int clusterID;
    float minDistance;    //distance to nearest centroid

    Point() {
        coords = (float*) malloc(dims * sizeof(float));
        clusterID = -1; //not yet assigned
        minDistance = __FLT_MAX__;
    }

    Point(float* c) {
        coords = c;
        clusterID = -1; //not yet assigned
        minDistance = __FLT_MAX__;
    }

    inline bool operator <(const Point& point) {
        //just sort by x-coordinate because want to compare what should be the same centroids
        return coords[0] < point.coords[0];
    }

    //computes euclidean distance between two dims-dimensional points
    float getDistance(Point* p) {
        float dist;
        
        /* Euclidean Distance Formula - could potentially optomize */
        for(int i = 0; i < dims; i++) {
            try {
                float temp = fabs(p->coords[i] - coords[i]);
                temp = temp * temp;
                dist += temp;
            } catch (std::exception& e) {
                std::cerr << "Exception 2: " << e.what() << '\n';
            }
        }
        return (float) sqrt(dist);
    }

    //assigns point to nearest centroid
    void assignNearestCentroid() {
        for(int i = 0;  i < num_clusters; i++) {
            Point p = centroidArr.at(i);
            float dist = getDistance(&p);
            if(dist < minDistance) {
                minDistance = dist;
                clusterID = i;
            }
        }
    }
};

/* boilerplate cuda error-checking code from a CUDA guide - used for debugging */
#define errorCheckCUDA(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"Assert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

//wrapper to start CUDA kernels
void assignClustersKernel(Point *devicePointArr, Point *deviceCentroidArr, int nPoints, int num_clusters, int dims);
void sumDimensionsKernel(Point *devicePointArr, int *deviceClusterSize, float* deviceSumDimensions, size_t pitch,
        int nPoints, int num_clusters, int dims);
void updateCentroidsKernel(Point *deviceCentroidArr , int *deviceClusterSize, float* deviceSumDimensions, size_t pitch,
        int nPoints, int num_clusters, int dims, bool* updated, double threshold);
void updateCentroidsSMKernel(Point *devicePointArr, Point *deviceCentroidArr , int *deviceClusterSize, float* deviceSumDimensions, size_t pitch,
        int nPoints, int num_clusters, int dims, bool* updated, double threshold);
void computeDValueKernel(Point *devicePointArr, Point *deviceCentroidArr, float* deviceDValues, int nPoints, int num_clusters, int dims);

/* Helper Functions */
extern float rand_float();
extern void intializeCentroidsRandom();

//returns true if oldCentroids and centroidArr are equivalent
extern bool converged(std::vector<Point> oldCentroids);
extern bool convergedCUDA(Point* oldCentroids, Point* hostCentroidArr);

/* Printing Functions */
extern void printPointArr();
extern void printPointAssignments();
extern void printCentroids();

#endif