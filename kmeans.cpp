#include <fstream>
#include <iostream>
#include <sstream>
#include <boost/program_options.hpp>
#include <iomanip>

#include "kmeans.h"

namespace po = boost::program_options;

/* Command Line Arguments */
int num_clusters, dims, max_num_iter, seed;
bool cFlag, gFlag, fFlag, pFlag;
double threshold;
std::string inputFileName, answerFileName;

/* Global Variables for Computation */
int nPoints;        //total number of points
float* DValues;      //stores DValues for corresponding points
Point* pointArr;       
std::vector<Point> centroidArr;

bool* updated = new bool[1];

/* Timing Variables */
int iter_to_converge;
float time_per_iter_in_ms;

/* Helper Functions + Printing */
float rand_float() {
    // Returns a float in [0.0, 1.0)
    return static_cast<float>(rand()) / static_cast<float>((long long) RAND_MAX+1);
}

void intializeCentroidsRandom() {
    for (int i = 0; i < num_clusters; i++) {
        int centroidIndex = rand_float() * nPoints;
        centroidArr.push_back(pointArr[centroidIndex]);
    }
}

//returns true if oldCentroids and centroidArr are equivalent - not used for convergence check anymore just testing
bool converged(std::vector<Point> *oldCentroids) {
    for(int i = 0; i < num_clusters; i++) {
        for(int j = 0; j < dims; j++) {
            float val1 = oldCentroids->at(i).coords[j];
            float val2 = centroidArr.at(i).coords[j];
            float difference = fabs(val1 - val2);
            if(difference > 0.0001f) {
                std::cout << val1 << "\n";
                std::cout << val2 << "\n";
                return false;
            }
        }
    }
    return true;
}

void printPointArr() {
    for(int i = 0; i < nPoints; i++) {
        printf("Point %d :", i+1);
        for(int j = 0; j < dims; j++) {
            try {
                printf(" %f", pointArr[i].coords[j]);
            }  catch (std::exception& e) {
                std::cerr << "Exception 3: " << e.what() << '\n';
            }
        }
        printf("\n");
    }
}

void printPointAssignments() {
    printf("clusters:");
    for(int i = 0; i < nPoints; i++) {
        printf(" %d", pointArr[i].clusterID);
    }
}

void printCentroids() {
    for (int clusterId = 0; clusterId < num_clusters; clusterId++){
        printf("%d ", clusterId);
        for (int d = 0; d < dims; d++) {
            try {
                printf("%lf ", centroidArr.at(clusterId).coords[d]);
            }  catch (std::exception& e) {
                std::cerr << "Exception 5: " << e.what() << '\n';
            }
        }
        printf("\n");
    }
}

//reads points from file specified by -i
void readInputFile() {
    using namespace std;
    
    ifstream file(inputFileName);
    string inputLine;

    getline(file, inputLine);
    try {
        nPoints = stoi(inputLine);
    } catch (std::exception& e) {
        std::cerr << "Exception: " << e.what() << '\n';
    }

    //initialize global vars
    pointArr = (Point*) malloc(nPoints * sizeof(struct Point));

    int index = 0;
    try {
        while (getline(file, inputLine)) {
            istringstream ss(inputLine);
            float* pointDimensions;    //stores dimensions for given point
            pointDimensions = (float*) malloc(dims * sizeof(float));
            
            string element;
            getline(ss, element, ' ');  //first element of each line is line number

            int i = 0;
            while(ss) {
                if (!getline(ss, element, ' ')) break;
                pointDimensions[i] = stod(element);
                i++;
            }

            pointArr[index] = Point(pointDimensions);
            index++;
        }
    }  catch (std::exception& e) {
        std::cerr << "Exception 4: " << e.what() << '\n';
    }
}

//read in centroids from answer file and compare to computed centroids - used for testing
void compareCentroids() {
    using namespace std;
    
    ifstream answerFile(answerFileName);
    string inputLine;

    vector<Point> answerCentroids;

    assert(answerFile.is_open());
    try {
        while (getline(answerFile, inputLine)) {    
            istringstream ss(inputLine);
            float* pointDimensions;    //stores dimensions for given point
            pointDimensions = (float*) malloc(dims * sizeof(float));

            string element;
            getline(ss, element, ' ');  //first element of each line is line number

            int i = 0;
            while(ss) {
                if (!getline(ss, element, ' ')) break;
                pointDimensions[i] = stod(element);
                i++;
            }

            answerCentroids.push_back(Point(pointDimensions));
        }
    }  catch (std::exception& e) {
        std::cerr << "Exception 6: " << e.what() << '\n';
    }

    //order can be different so have to sort
    std::sort(centroidArr.begin(), centroidArr.end());
    std::sort(answerCentroids.begin(), answerCentroids.end());

    std::cout << "Answer Sorted: \n";
    try {
        for (std::vector<Point>::const_iterator i = answerCentroids.begin(); i != answerCentroids.end(); ++i) {
                for (int j = 0; j < dims; j++) {
                    std::cout << (*i).coords[j] << ' ';
                }
                std::cout << "\n";
        }
    }  catch (std::exception& e) {
        std::cerr << "Exception 7: " << e.what() << '\n';
    }
    std::cout << "Computed Sorted: \n";
    try {
        for (std::vector<Point>::const_iterator i = centroidArr.begin(); i != centroidArr.end(); ++i) {
               for (int j = 0; j < dims; j++) {
                    std::cout << (*i).coords[j] << ' ';
                }
                std::cout << "\n";
        }
    }  catch (std::exception& e) {
        std::cerr << "Exception 8: " << e.what() << '\n';
    }

    //compares equality of computed centroids and the answer within 10^-4 of each dimension of each point
    if(converged(&answerCentroids)) {
        std::cout << "Correct Centroids!\n";
    } else {
        std::cout << "Incorrect Centroids!\n";
    }
}

/* Sequential K-Means - Lloyd's Algorithm */
void sequentialKMeans() {
    using namespace std::chrono;

    // float assignToClustersTime, sumDimensionsTime, updateCentroidsTime;

    auto startTime = high_resolution_clock::now();
    intializeCentroidsRandom();   //randomly assign centroids

    bool done = false;
    while(!done) {
        //book-keeping
        std::vector<Point> oldCentroids(centroidArr);   //deep copy
        iter_to_converge++;
        updated[0] = false;

        // auto startAssignToClusters = high_resolution_clock::now();
        //assign points to clusters
        for(int i = 0; i < nPoints; i++) {
            pointArr[i].assignNearestCentroid();
        }
        // auto endAssignToClusters = high_resolution_clock::now();
        // duration<double, std::milli> elapsedAssignToClusters = endAssignToClusters - startAssignToClusters;
        // assignToClustersTime += elapsedAssignToClusters.count();

        // auto startSumDimensions = high_resolution_clock::now();

        std::vector<int> clusterSize(num_clusters, 0);     //int # of points within each cluster
        std::vector<std::vector<float>> sumDimensions;     //sums of respective x, y, z, etc. coordinates for points in a cluster
        for(int j = 0; j < num_clusters; j++) {
            std::vector<float> temp (dims, 0.0f);
            sumDimensions.push_back(temp);
        }

        //sum coordinates of points in a cluster + size of clusters
        try {
            for(int i = 0; i < nPoints; i++) {
                clusterSize.at(pointArr[i].clusterID) += 1;
                for(int j = 0; j < dims; j++) {
                    sumDimensions.at(pointArr[i].clusterID).at(j) += pointArr[i].coords[j];
                }
                pointArr[i].minDistance = __FLT_MAX__;  //reset min distance
            }
        } catch (std::exception& e) {
            std::cerr << "Exception 9: " << e.what() << '\n';
        }

        // auto endSumDimensions = high_resolution_clock::now();
        // duration<double, std::milli> elapsedSumDimensions = endSumDimensions - startSumDimensions;
        // sumDimensionsTime += elapsedSumDimensions.count();

        //recompute centroids based on mean of point coordinates
        // auto startUpdateCentroids = high_resolution_clock::now();

        try {
            for(int i = 0; i < num_clusters; i++) {
                //don't recompute empty clusters
                if(clusterSize.at(i) != 0) {
                    for(int j = 0; j < dims; j++) {
                        //this is our convergence check
                        float meanVal = sumDimensions.at(i).at(j) / float(clusterSize.at(i));
                        if(fabs(centroidArr.at(i).coords[j] - meanVal) > float(threshold)) {
                            updated[0] = true;
                        }
                        centroidArr.at(i).coords[j] = sumDimensions.at(i).at(j) / float(clusterSize.at(i));
                    }
                }
            }
        } catch (std::exception& e) {
            std::cerr << "Exception 10: " << e.what() << '\n';
        }

        // auto endUpdateCentroids = high_resolution_clock::now();
        // duration<double, std::milli> elapsedUpdateCentroids = endUpdateCentroids - startUpdateCentroids;
        // updateCentroidsTime += elapsedUpdateCentroids.count();

        done = (iter_to_converge > max_num_iter) || !updated[0];
    }
    auto endTime = high_resolution_clock::now();
    duration<double, std::milli> elapsed = endTime - startTime;
    time_per_iter_in_ms = elapsed.count() / iter_to_converge;
    
    // printf("Assign Clusters: %lf\n", assignToClustersTime / iter_to_converge);
    // printf("Sum Dimensions: %lf\n", sumDimensionsTime / iter_to_converge);
    // printf("Assign Centroids: %lf\n", updateCentroidsTime / iter_to_converge);
}

/* Basic CUDA Implementation */
void parallelKMeansCUDA() {
    //cudaEvent variables for timing memory transactions, etc.
    // float assignToClustersTime, sumDimensionsTime, updateCentroidsTime, dataTransferTime;
    // cudaEvent_t assignToClustersStart, assignToClustersEnd;
    // cudaEventCreate(&assignToClustersStart);
    // cudaEventCreate(&assignToClustersEnd);
    // cudaEvent_t sumDimensionsTimeStart, sumDimensionsTimeEnd;
    // cudaEventCreate(&sumDimensionsTimeStart);
    // cudaEventCreate(&sumDimensionsTimeEnd);
    // cudaEvent_t updateCentroidsStart, updateCentroidsEnd;
    // cudaEventCreate(&updateCentroidsStart);
    // cudaEventCreate(&updateCentroidsEnd);
    // cudaEvent_t dataTransferStart, dataTransferEnd;
    // cudaEventCreate(&dataTransferStart);
    // cudaEventCreate(&dataTransferEnd);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);

    intializeCentroidsRandom();   //randomly assign centroids

    //book-keeping
    Point* oldCentroids = (Point*) malloc(num_clusters * sizeof(struct Point));

    //setup device arrays/memory
	Point *devicePointArr, *tempPointArr;
    Point *deviceCentroidArr, *tempCentroidArr;
    int* deviceClusterSize;     //int # of points within each cluster
    float* deviceSumDimensions; //actually a 2d array - holds sums of respective x, y, z, etc. coordinates for points in a cluster
    bool* deviceUpdated;

    // cudaEventRecord(dataTransferStart);

	cudaMalloc((void**) &deviceClusterSize, num_clusters * sizeof(int));
    size_t devicePitch;   //save the pitch value so we can correctly access array elements
	cudaMallocPitch((void**) &deviceSumDimensions, &devicePitch,  dims * sizeof(float), num_clusters);
    cudaMalloc((void**) &deviceUpdated, sizeof(bool*));

    //setup host memory
	Point *hostCentroidArr = (Point*) malloc(num_clusters * sizeof(struct Point));
    std::copy(centroidArr.begin(), centroidArr.end(), hostCentroidArr);
    tempPointArr = (Point*) malloc(nPoints * sizeof(struct Point));
    tempCentroidArr = (Point*) malloc(num_clusters * sizeof(struct Point));
	int *hostClusterSize = (int*) malloc(num_clusters * sizeof(int));
    float* hostSumDimensions = (float*) malloc(num_clusters * dims * sizeof(float));

    //copy host memory to device memory for use by kernels have to use a temp array to copy over pointers
    for(int i = 0; i < nPoints; i++) {
        float* tempCoords;
        cudaMalloc((void**) &tempCoords, dims * sizeof(float));
        cudaMemcpy(tempCoords, pointArr[i].coords, dims * sizeof(float), cudaMemcpyHostToDevice);

        tempPointArr[i] = Point(tempCoords);
    }
    cudaMalloc((void**) &devicePointArr, nPoints * sizeof(struct Point));
    cudaMemcpy(devicePointArr, tempPointArr, nPoints * sizeof(struct Point), cudaMemcpyHostToDevice);

    for(int i = 0; i < num_clusters; i++) {
        float* tempCoords;
        cudaMalloc((void**) &tempCoords, dims * sizeof(float));
        cudaMemcpy(tempCoords, hostCentroidArr[i].coords, dims * sizeof(float), cudaMemcpyHostToDevice);

        tempCentroidArr[i] = Point(tempCoords);
    }
    cudaMalloc((void**) &deviceCentroidArr, num_clusters * sizeof(struct Point));
    cudaMemcpy(deviceCentroidArr, tempCentroidArr, num_clusters * sizeof(struct Point), cudaMemcpyHostToDevice);

	cudaMemcpy(deviceClusterSize, hostClusterSize, num_clusters * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy2D(deviceSumDimensions, devicePitch, hostSumDimensions, dims * sizeof(float), dims * sizeof(float), num_clusters, cudaMemcpyHostToDevice);

    // cudaEventRecord(dataTransferEnd);
    // float temp = 0;
    // cudaEventSynchronize(dataTransferEnd);
    // cudaEventElapsedTime(&temp, dataTransferStart, dataTransferEnd);
    // dataTransferTime += temp;

    bool done = false;
    while(!done) {
        iter_to_converge++;

        oldCentroids = hostCentroidArr;

        //start kernel to assign points to nearest centroid
        // cudaEventRecord(assignToClustersStart);
        assignClustersKernel(devicePointArr, deviceCentroidArr, nPoints, num_clusters, dims);
        cudaDeviceSynchronize();
        // cudaEventRecord(assignToClustersEnd);
        // temp = 0;
        // cudaEventSynchronize(assignToClustersEnd);
        // cudaEventElapsedTime(&temp, assignToClustersStart, assignToClustersEnd);
        // assignToClustersTime += temp;

        //copy back new point values (with updated centroidID) from device to host memory
        // cudaEventRecord(dataTransferStart);
        cudaMemcpy(pointArr, devicePointArr, nPoints * sizeof(struct Point), cudaMemcpyDeviceToHost);

        //clear summation arrays
        cudaMemset(deviceClusterSize, 0, num_clusters * sizeof(int));
        cudaMemset2D(deviceSumDimensions, devicePitch, 0, dims * sizeof(float), num_clusters);
        
        // cudaEventRecord(dataTransferEnd);
        // temp = 0;
        // cudaEventSynchronize(dataTransferEnd);
        // cudaEventElapsedTime(&temp, dataTransferStart, dataTransferEnd);
        // dataTransferTime += temp;

        //sum dimensions of points and cluster sizes
        // cudaEventRecord(sumDimensionsTimeStart);
        sumDimensionsKernel(devicePointArr, deviceClusterSize, deviceSumDimensions, devicePitch, nPoints, num_clusters, dims);
        cudaDeviceSynchronize();
        // cudaEventRecord(sumDimensionsTimeEnd);
        // temp = 0;
        // cudaEventSynchronize(sumDimensionsTimeEnd);
        // cudaEventElapsedTime(&temp, sumDimensionsTimeStart, sumDimensionsTimeEnd);
        // sumDimensionsTime += temp;

        //copy back summed values
        // cudaEventRecord(dataTransferStart);
        cudaMemcpy(hostClusterSize, deviceClusterSize, num_clusters * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy2D(hostSumDimensions, dims * sizeof(float), deviceSumDimensions, devicePitch, dims * sizeof(float), num_clusters, cudaMemcpyDeviceToHost);
        // cudaEventRecord(dataTransferEnd);
        // temp = 0;
        // cudaEventSynchronize(dataTransferEnd);
        // cudaEventElapsedTime(&temp, dataTransferStart, dataTransferEnd);
        // dataTransferTime += temp;

        //compute cluster means and update centroids
        // cudaEventRecord(updateCentroidsStart);
        updated[0] = false;
        cudaMemcpy(deviceUpdated, updated, sizeof(bool*), cudaMemcpyHostToDevice);
        updateCentroidsKernel(deviceCentroidArr, deviceClusterSize, deviceSumDimensions, devicePitch, nPoints, num_clusters, dims, deviceUpdated, threshold);
        cudaDeviceSynchronize();
        // cudaEventRecord(updateCentroidsEnd);
        // temp = 0;
        // cudaEventSynchronize(updateCentroidsEnd);
        // cudaEventElapsedTime(&temp, updateCentroidsStart, updateCentroidsEnd);
        // updateCentroidsTime += temp;
        
        //copy back updated centroids from device to host
        // cudaEventRecord(dataTransferStart);
        cudaMemcpy(hostCentroidArr, deviceCentroidArr, num_clusters * sizeof(struct Point), cudaMemcpyDeviceToHost);
        for(int i = 0; i < num_clusters; i++) {
            float* coords = hostCentroidArr[i].coords;
            hostCentroidArr[i].coords = (float*) malloc(dims * sizeof(float));

            cudaMemcpy(hostCentroidArr[i].coords, coords, dims * sizeof(float), cudaMemcpyDeviceToHost);
        }
        
        //copy back convergence flag
        cudaMemcpy(updated, deviceUpdated, sizeof(bool*), cudaMemcpyDeviceToHost);

        // cudaEventRecord(dataTransferEnd);
        // temp = 0;
        // cudaEventSynchronize(dataTransferEnd);
        // cudaEventElapsedTime(&temp, dataTransferStart, dataTransferEnd);
        // dataTransferTime += temp;

        done = iter_to_converge > max_num_iter || !updated[0];
    }
    // cudaEventRecord(dataTransferStart);

    //free malloced device memory
    cudaFree(devicePointArr);
	cudaFree(deviceCentroidArr);
    cudaFree(deviceSumDimensions);
    cudaFree(deviceClusterSize);

    //free malloced host memory - copy back to vector centroidArr beforehand
    std::vector<Point> centroids;
    for(int i = 0; i < num_clusters; i++) {
        centroids.push_back(hostCentroidArr[i]);
    }
    centroidArr = centroids;
	free(hostCentroidArr);
    free(hostClusterSize);
    free(hostSumDimensions);
    free(tempPointArr);
    free(tempCentroidArr);

    // cudaEventRecord(dataTransferEnd);
    // temp = 0;
    // cudaEventSynchronize(dataTransferEnd);
    // cudaEventElapsedTime(&temp, dataTransferStart, dataTransferEnd);
    // dataTransferTime += temp;

    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float elapsed = 0;  //in milliseconds
    cudaEventElapsedTime(&elapsed, start, end);

    time_per_iter_in_ms = elapsed / iter_to_converge;

    // printf("Assign Clusters: %lf\n", assignToClustersTime / iter_to_converge);
    // printf("Sum Dimensions: %lf\n", sumDimensionsTime / iter_to_converge);
    // printf("Assign Centroids: %lf\n", updateCentroidsTime / iter_to_converge);
    // printf("Data Transfer Time: %lf\n", dataTransferTime);
}


/* CUDA Implementation with Shared Memory */
void parallelKMeansCUDASharedMem() {
    //cudaEvent variables for timing memory transactions, etc.

    // float assignToClustersTime, sumDimensionsTime, dataTransferTime;
    // cudaEvent_t assignToClustersStart, assignToClustersEnd;
    // cudaEventCreate(&assignToClustersStart);
    // cudaEventCreate(&assignToClustersEnd);
    // cudaEvent_t sumDimensionsStart, sumDimensionsEnd;
    // cudaEventCreate(&sumDimensionsStart);
    // cudaEventCreate(&sumDimensionsEnd);
    // cudaEvent_t dataTransferStart, dataTransferEnd;
    // cudaEventCreate(&dataTransferStart);
    // cudaEventCreate(&dataTransferEnd);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);

    intializeCentroidsRandom();   //randomly assign centroids

    //book-keeping
    Point* oldCentroids = (Point*) malloc(num_clusters * sizeof(struct Point));

    //setup device arrays/memory
	Point *devicePointArr, *tempPointArr;
    Point *deviceCentroidArr, *tempCentroidArr;
    int* deviceClusterSize;     //int # of points within each cluster
    float* deviceSumDimensions; //actually a 2d array - holds sums of respective x, y, z, etc. coordinates for points in a cluster
    bool* deviceUpdated;

    // cudaEventRecord(dataTransferStart);
   
	cudaMalloc((void**) &deviceClusterSize, num_clusters * sizeof(int));
    size_t devicePitch;   //save the pitch value so we can correctly access array elements
	cudaMallocPitch((void**) &deviceSumDimensions, &devicePitch,  dims * sizeof(float), num_clusters);
    cudaMalloc((void**) &deviceUpdated, sizeof(bool*));

    //setup host memory
	Point *hostCentroidArr = (Point*) malloc(num_clusters * sizeof(struct Point));
    std::copy(centroidArr.begin(), centroidArr.end(), hostCentroidArr);
    tempPointArr = (Point*) malloc(nPoints * sizeof(struct Point));
    tempCentroidArr = (Point*) malloc(num_clusters * sizeof(struct Point));
	int *hostClusterSize = (int*) malloc(num_clusters * sizeof(int));
    float* hostSumDimensions = (float*) malloc(num_clusters * dims * sizeof(float));

    //copy host memory to device memory for use by kernels
    for(int i = 0; i < nPoints; i++) {
        float* tempCoords;
        cudaMalloc((void**) &tempCoords, dims * sizeof(float));
        cudaMemcpy(tempCoords, pointArr[i].coords, dims * sizeof(float), cudaMemcpyHostToDevice);

        tempPointArr[i] = Point(tempCoords);
    }
    cudaMalloc((void**) &devicePointArr, nPoints * sizeof(struct Point));
    cudaMemcpy(devicePointArr, tempPointArr, nPoints * sizeof(struct Point), cudaMemcpyHostToDevice);

    for(int i = 0; i < num_clusters; i++) {
        float* tempCoords;
        cudaMalloc((void**) &tempCoords, dims * sizeof(float));
        cudaMemcpy(tempCoords, hostCentroidArr[i].coords, dims * sizeof(float), cudaMemcpyHostToDevice);

        tempCentroidArr[i] = Point(tempCoords);
    }
    cudaMalloc((void**) &deviceCentroidArr, num_clusters * sizeof(struct Point));
    cudaMemcpy(deviceCentroidArr, tempCentroidArr, num_clusters * sizeof(struct Point), cudaMemcpyHostToDevice);

	cudaMemcpy(deviceClusterSize, hostClusterSize, num_clusters * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy2D(deviceSumDimensions, devicePitch, hostSumDimensions, dims * sizeof(float), dims * sizeof(float), num_clusters, cudaMemcpyHostToDevice);

    // cudaEventRecord(dataTransferEnd);
    // float temp = 0;
    // cudaEventSynchronize(dataTransferEnd);
    // cudaEventElapsedTime(&temp, dataTransferStart, dataTransferEnd);
    // dataTransferTime += temp;

    bool done = false;
    while(!done) {
        iter_to_converge++;

        oldCentroids = hostCentroidArr;

        //start kernel to assign points to nearest centroid - same as first CUDA implementation
        // cudaEventRecord(assignToClustersStart);
        assignClustersKernel(devicePointArr, deviceCentroidArr, nPoints, num_clusters, dims);
        cudaDeviceSynchronize();
        // cudaEventRecord(assignToClustersEnd);
        // temp = 0;
        // cudaEventSynchronize(assignToClustersEnd);
        // cudaEventElapsedTime(&temp, assignToClustersStart, assignToClustersEnd);
        // assignToClustersTime += temp;

        //copy back new point values (with updated centroidID) from device to host memory
        // cudaEventRecord(dataTransferStart);
        cudaMemcpy(pointArr, devicePointArr, nPoints * sizeof(struct Point), cudaMemcpyDeviceToHost);

        //clear summation arrays
        cudaMemset(deviceClusterSize, 0, num_clusters * sizeof(int));
        cudaMemset2D(deviceSumDimensions, devicePitch, 0, dims * sizeof(float), num_clusters);

        // cudaEventRecord(dataTransferEnd);
        // temp = 0;
        // cudaEventSynchronize(dataTransferEnd);
        // cudaEventElapsedTime(&temp, dataTransferStart, dataTransferEnd);
        // dataTransferTime += temp;

        //sum dimensions of points and cluster sizes using shared memory
        // cudaEventRecord(sumDimensionsStart);
        updated[0] = false;
        cudaMemcpy(deviceUpdated, updated, sizeof(bool*), cudaMemcpyHostToDevice);
        updateCentroidsSMKernel(devicePointArr, deviceCentroidArr, deviceClusterSize, deviceSumDimensions, devicePitch, nPoints, num_clusters, 
                                dims, deviceUpdated, threshold);
        cudaDeviceSynchronize();
        // cudaEventRecord(sumDimensionsEnd);
        // temp = 0;
        // cudaEventSynchronize(sumDimensionsEnd);
        // cudaEventElapsedTime(&temp, sumDimensionsStart, sumDimensionsEnd);
        // sumDimensionsTime += temp;

        //copy back updated centroids from device to host
        // cudaEventRecord(dataTransferStart);
        cudaMemcpy(hostCentroidArr, deviceCentroidArr, num_clusters * sizeof(struct Point), cudaMemcpyDeviceToHost);
        for(int i = 0; i < num_clusters; i++) {
            float* coords = hostCentroidArr[i].coords;
            hostCentroidArr[i].coords = (float*) malloc(dims * sizeof(float));

            cudaMemcpy(hostCentroidArr[i].coords, coords, dims * sizeof(float), cudaMemcpyDeviceToHost);
        }
        
        //copy back convergence flag
        cudaMemcpy(updated, deviceUpdated, sizeof(bool*), cudaMemcpyDeviceToHost);

        // cudaEventRecord(dataTransferEnd);
        // temp = 0;
        // //cudaEventSynchronize(dataTransferEnd);
        // cudaEventElapsedTime(&temp, dataTransferStart, dataTransferEnd);
        // dataTransferTime += temp;


        done = iter_to_converge > max_num_iter || !updated[0];
    }

    // cudaEventRecord(dataTransferStart);

    //free malloced device memory
    cudaFree(devicePointArr);
	cudaFree(deviceCentroidArr);
    cudaFree(deviceSumDimensions);
    cudaFree(deviceClusterSize);

    //free malloced host memory - copy back to vector centroidArr beforehand
    std::vector<Point> centroids;
    for(int i = 0; i < num_clusters; i++) {
        centroids.push_back(hostCentroidArr[i]);
    }
    centroidArr = centroids;
	free(hostCentroidArr);
    free(hostClusterSize);
    free(hostSumDimensions);
    free(tempPointArr);
    free(tempCentroidArr);

    // cudaEventRecord(dataTransferEnd);
    // temp = 0;
    // cudaEventSynchronize(dataTransferEnd);
    // cudaEventElapsedTime(&temp, dataTransferStart, dataTransferEnd);
    // dataTransferTime += temp;

    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float elapsed = 0;
    cudaEventElapsedTime(&elapsed, start, end);

    time_per_iter_in_ms = elapsed / iter_to_converge;

    // printf("Assign Clusters: %lf\n", assignToClustersTime / iter_to_converge);
    // printf("Sum Dimensions: %lf\n", sumDimensionsTime / iter_to_converge);
    // printf("Data Transfer Time: %lf\n", dataTransferTime);
}

/* K-means++ Implementation */

void resetDValue() {
    for(int i = 0; i < nPoints; i++) {
        DValues[i] = __FLT_MAX__;
    }
}

//just a wrapper to call the kernel init function
void computeDValues(Point * devicePointArr, Point * deviceCentroidArr, float* deviceDValues, int currNumCentroids) {
    //start kernel to compute D(x) for all points.
    computeDValueKernel(devicePointArr, deviceCentroidArr, deviceDValues, nPoints, currNumCentroids, dims);
    cudaDeviceSynchronize();

    //copy back new D-Values for points from device to host memory
    cudaMemcpy(DValues, deviceDValues, nPoints * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
}

void kMeansPlusPlusInitializeCentroids() {
    // float dataTransferTime;
    // cudaEvent_t dataTransferStart, dataTransferEnd;
    // cudaEventCreate(&dataTransferStart);
    // cudaEventCreate(&dataTransferEnd);

    int firstIndex = (int) (rand_float() * nPoints);
    int currNumCentroids = 1;
    Point firstCentroid = pointArr[firstIndex];
    centroidArr.push_back(firstCentroid);

    //initialize DValues
    DValues = (float*) malloc(nPoints * sizeof(float));
    resetDValue();

    //setup device arrays/memory
    Point *devicePointArr, *tempPointArr;
    Point *deviceCentroidArr, *tempCentroidArr;
    float *deviceDValues;

    // cudaEventRecord(dataTransferStart);

    cudaMalloc((void**) &deviceDValues, nPoints * sizeof(float));
    cudaMemcpy(deviceDValues, DValues, nPoints * sizeof(float), cudaMemcpyHostToDevice);

    //setup host memory
    Point *hostCentroidArr;
    tempPointArr = (Point*) malloc(nPoints * sizeof(struct Point));
    
    //copy host memory to device memory for use by kernels
    for(int i = 0; i < nPoints; i++) {
        float* tempCoords;
        cudaMalloc((void**) &tempCoords, dims * sizeof(float));
        cudaMemcpy(tempCoords, pointArr[i].coords, dims * sizeof(float), cudaMemcpyHostToDevice);

        tempPointArr[i] = Point(tempCoords);
    }
    cudaMalloc((void**) &devicePointArr, nPoints * sizeof(struct Point));
    cudaMemcpy(devicePointArr, tempPointArr, nPoints * sizeof(struct Point), cudaMemcpyHostToDevice);

    // cudaEventRecord(dataTransferEnd);
    // float temp = 0;
    // cudaEventSynchronize(dataTransferEnd);
    // cudaEventElapsedTime(&temp, dataTransferStart, dataTransferEnd);
    // dataTransferTime += temp;
    
    int index = 0;
    while (centroidArr.size() < num_clusters) {
        //have to copy the new centroidArr to device memory every iteration - very expensive
        // cudaEventRecord(dataTransferStart);

        hostCentroidArr = (Point*) malloc(currNumCentroids * sizeof(struct Point));
        tempCentroidArr = (Point*) malloc(currNumCentroids * sizeof(struct Point));
        std::copy(centroidArr.begin(), centroidArr.end(), hostCentroidArr);
        for(int i = 0; i < currNumCentroids; i++) {
            float* tempCoords;
            cudaMalloc((void**) &tempCoords, dims * sizeof(float));
            cudaMemcpy(tempCoords, hostCentroidArr[i].coords, dims * sizeof(float), cudaMemcpyHostToDevice);
            tempCentroidArr[i] = Point(tempCoords);
        }
        cudaMalloc((void**) &deviceCentroidArr, currNumCentroids * sizeof(struct Point));
        cudaMemcpy(deviceCentroidArr, tempCentroidArr, currNumCentroids * sizeof(struct Point), cudaMemcpyHostToDevice);

        // cudaEventRecord(dataTransferEnd);
        // temp = 0;
        // cudaEventSynchronize(dataTransferEnd);
        // cudaEventElapsedTime(&temp, dataTransferStart, dataTransferEnd);
        // dataTransferTime += temp;
        
        //calculate D-values for all points
        computeDValues(devicePointArr, deviceCentroidArr, deviceDValues, currNumCentroids);

        // Choose a new initial centroid
        float totalDistance = 0.0;
        for (int i = 0; i < nPoints; i++) {
            float dValue = DValues[i];
            totalDistance += (dValue * dValue);
        }
        float target = rand_float() * totalDistance;
        float dist = 0.0;
        for (int i = 0; i < nPoints; i++) {
            float dValue = DValues[i];
           
            dist += (dValue * dValue);
            if (target < dist) {
                Point c = pointArr[i];
                centroidArr.push_back(c);
                currNumCentroids++;
                break;
            }
        }
        resetDValue();                  //reset to flt max so can recompute
        //cudaFree(deviceCentroidArr);   //recomputes centroids every loop
        //free(hostCentroidArr);
        //free(tempCentroidArr);
    }
    //cudaFree(devicePointArr);    
    //free(tempPointArr);

    // printf("Data Transfer Time: %lf\n", dataTransferTime);
}

void kMeansPlusPlus() {
    //timing variables
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    // float dataTransferTime;
    // cudaEvent_t dataTransferStart, dataTransferEnd;
    // cudaEventCreate(&dataTransferStart);
    // cudaEventCreate(&dataTransferEnd);

    // cudaEvent_t kMeansIntializeStart, kMeansIntializeEnd;
    // cudaEventCreate(&kMeansIntializeStart);
    // cudaEventCreate(&kMeansIntializeEnd);

    cudaEventRecord(start);

    //choose initial centroids according to kmeans++
    // cudaEventRecord(kMeansIntializeStart);
    kMeansPlusPlusInitializeCentroids();
    // cudaEventRecord(kMeansIntializeEnd);
    // float time = 0;  //in milliseconds
    // cudaEventSynchronize(kMeansIntializeEnd);
    // cudaEventElapsedTime(&time, kMeansIntializeStart, kMeansIntializeEnd);
    
    //rest of solution is the same as first GPU implementation

    //book-keeping
    Point* oldCentroids = (Point*) malloc(num_clusters * sizeof(struct Point));

    //setup device arrays/memory
	Point *devicePointArr, *tempPointArr;
    Point *deviceCentroidArr, *tempCentroidArr;
    int* deviceClusterSize;     //int # of points within each cluster
    float* deviceSumDimensions; //actually a 2d array - holds sums of respective x, y, z, etc. coordinates for points in a cluster
    bool* deviceUpdated;

    // cudaEventRecord(dataTransferStart);
   
	cudaMalloc((void**) &deviceClusterSize, num_clusters * sizeof(int));
    size_t devicePitch;   //save the pitch value so we can correctly access array elements
	cudaMallocPitch((void**) &deviceSumDimensions, &devicePitch,  dims * sizeof(float), num_clusters);
    cudaMalloc((void**) &deviceUpdated, sizeof(bool*));

    //setup host memory
	Point *hostCentroidArr = (Point*) malloc(num_clusters * sizeof(struct Point));
    std::copy(centroidArr.begin(), centroidArr.end(), hostCentroidArr);
    tempPointArr = (Point*) malloc(nPoints * sizeof(struct Point));
    tempCentroidArr = (Point*) malloc(num_clusters * sizeof(struct Point));
	int *hostClusterSize = (int*) malloc(num_clusters * sizeof(int));
    float* hostSumDimensions = (float*) malloc(num_clusters * dims * sizeof(float));

    //copy host memory to device memory for use by kernels have to use a temp array to copy over pointers
    for(int i = 0; i < nPoints; i++) {
        float* tempCoords;
        cudaMalloc((void**) &tempCoords, dims * sizeof(float));
        cudaMemcpy(tempCoords, pointArr[i].coords, dims * sizeof(float), cudaMemcpyHostToDevice);

        tempPointArr[i] = Point(tempCoords);
    }
    cudaMalloc((void**) &devicePointArr, nPoints * sizeof(struct Point));
    cudaMemcpy(devicePointArr, tempPointArr, nPoints * sizeof(struct Point), cudaMemcpyHostToDevice);

    for(int i = 0; i < num_clusters; i++) {
        float* tempCoords;
        cudaMalloc((void**) &tempCoords, dims * sizeof(float));
        cudaMemcpy(tempCoords, hostCentroidArr[i].coords, dims * sizeof(float), cudaMemcpyHostToDevice);

        tempCentroidArr[i] = Point(tempCoords);
    }
    cudaMalloc((void**) &deviceCentroidArr, num_clusters * sizeof(struct Point));
    cudaMemcpy(deviceCentroidArr, tempCentroidArr, num_clusters * sizeof(struct Point), cudaMemcpyHostToDevice);

	cudaMemcpy(deviceClusterSize, hostClusterSize, num_clusters * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy2D(deviceSumDimensions, devicePitch, hostSumDimensions, dims * sizeof(float), dims * sizeof(float), num_clusters, cudaMemcpyHostToDevice);

    // cudaEventRecord(dataTransferEnd);
    // float temp = 0;
    // cudaEventSynchronize(dataTransferEnd);
    // cudaEventElapsedTime(&temp, dataTransferStart, dataTransferEnd);
    // dataTransferTime += temp;

    bool done = false;
    while(!done) {
        iter_to_converge++;

        oldCentroids = hostCentroidArr;

        //start kernel to assign points to nearest centroid
        assignClustersKernel(devicePointArr, deviceCentroidArr, nPoints, num_clusters, dims);
        cudaDeviceSynchronize();

        // cudaEventRecord(dataTransferStart);

        //copy back new point values (with updated centroidID) from device to host memory
        cudaMemcpy(pointArr, devicePointArr, nPoints * sizeof(struct Point), cudaMemcpyDeviceToHost);

        //clear summation arrays
        cudaMemset(deviceClusterSize, 0, num_clusters * sizeof(int));
        cudaMemset2D(deviceSumDimensions, devicePitch, 0, dims * sizeof(float), num_clusters);

        // cudaEventRecord(dataTransferEnd);
        // temp = 0;
        // cudaEventSynchronize(dataTransferEnd);
        // cudaEventElapsedTime(&temp, dataTransferStart, dataTransferEnd);
        // dataTransferTime += temp;

        //sum dimensions of points and cluster sizes
        sumDimensionsKernel(devicePointArr, deviceClusterSize, deviceSumDimensions, devicePitch, nPoints, num_clusters, dims);
        cudaDeviceSynchronize();

        // cudaEventRecord(dataTransferStart);

        //copy back summed values
        cudaMemcpy(hostClusterSize, deviceClusterSize, num_clusters * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy2D(hostSumDimensions, dims * sizeof(float), deviceSumDimensions, devicePitch, dims * sizeof(float), num_clusters, cudaMemcpyDeviceToHost);

        // cudaEventRecord(dataTransferEnd);
        // temp = 0;
        // cudaEventSynchronize(dataTransferEnd);
        // cudaEventElapsedTime(&temp, dataTransferStart, dataTransferEnd);
        // dataTransferTime += temp;

        //sum dimensions of points and cluster sizes
        updated[0] = false;
        cudaMemcpy(deviceUpdated, updated, sizeof(bool*), cudaMemcpyHostToDevice);
        updateCentroidsKernel(deviceCentroidArr, deviceClusterSize, deviceSumDimensions, devicePitch, nPoints, num_clusters, dims, deviceUpdated, threshold);
        cudaDeviceSynchronize();
        
        // cudaEventRecord(dataTransferStart);
        
        cudaMemcpy(hostCentroidArr, deviceCentroidArr, num_clusters * sizeof(struct Point), cudaMemcpyDeviceToHost);

        //copy back updated centroids from device to host
        cudaMemcpy(hostCentroidArr, deviceCentroidArr, num_clusters * sizeof(struct Point), cudaMemcpyDeviceToHost);
        for(int i = 0; i < num_clusters; i++) {
            float* coords = hostCentroidArr[i].coords;
            hostCentroidArr[i].coords = (float*) malloc(dims * sizeof(float));

            cudaMemcpy(hostCentroidArr[i].coords, coords, dims * sizeof(float), cudaMemcpyDeviceToHost);
        }
        
        //copy back convergence flag
        cudaMemcpy(updated, deviceUpdated, sizeof(bool*), cudaMemcpyDeviceToHost);

        // cudaEventRecord(dataTransferEnd);
        // temp = 0;
        // cudaEventSynchronize(dataTransferEnd);
        // cudaEventElapsedTime(&temp, dataTransferStart, dataTransferEnd);
        // dataTransferTime += temp;

        done = iter_to_converge > max_num_iter || !updated[0];
    }
    
    // cudaEventRecord(dataTransferStart);

    //free malloced device memory
    cudaFree(devicePointArr);
	cudaFree(deviceCentroidArr);
    cudaFree(deviceSumDimensions);
    cudaFree(deviceClusterSize);

    //free malloced host memory - copy back to vector centroidArr beforehand
    std::vector<Point> centroids;
    for(int i = 0; i < num_clusters; i++) {
        centroids.push_back(hostCentroidArr[i]);
    }
    centroidArr = centroids;
	free(hostCentroidArr);
    free(hostClusterSize);
    free(hostSumDimensions);
    free(tempPointArr);
    free(tempCentroidArr);

    // cudaEventRecord(dataTransferEnd);
    // temp = 0;
    // cudaEventSynchronize(dataTransferEnd);
    // cudaEventElapsedTime(&temp, dataTransferStart, dataTransferEnd);
    // dataTransferTime += temp;

    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float elapsed = 0;  //in milliseconds
    cudaEventElapsedTime(&elapsed, start, end);
    
    time_per_iter_in_ms = elapsed / iter_to_converge;

    // printf("Time to Initialize: %lf\n", time);
    // printf("Data Transfer Time: %lf\n", dataTransferTime);
}

int main(int argc, char* argv[]) {
    using namespace std;
    using namespace std::chrono;

    // auto startTime = high_resolution_clock::now();

    // cudaEvent_t start, end;
    // cudaEventCreate(&start);
    // cudaEventCreate(&end);
    // cudaEventRecord(start);

    //basic command line parsing with boost - https://www.boost.org/doc/libs/1_75_0/doc/html/program_options/tutorial.html#id-1.3.32.4.3
    //takes arguments of the form: -command value (i.e. -k 16)
    try {
        //set which options can be accepted
        po::options_description desc("Options");
        desc.add_options()
            ("k,k", po::value<int>(), "the number of clusters")
            ("d,d", po::value<int>(), "the dimension of the points")
            ("i,i", po::value<string>(), "the input filename")
            ("m,m", po::value<int>(), "the maximum number of iterations")
            ("t,t", po::value<double>(), "the threshold for convergence test")
            ("c,c", "output the centroids of all clusters")
            ("s,s", po::value<int>(), "the seed for rand()")
            ("g,g", "enable the GPU implementation")
            ("f,f", "enable the shared-memory GPU implementation")
            ("p,p", "enable the Kmeans++ implementation")
            ("a,a", po::value<string>()->default_value(""), "added for testing - filename containing correct output");

        //parse command line arguments
        po::variables_map args;
        po::store(po::parse_command_line(argc, argv, desc), args);
        po::notify(args);

        if(args.count("k")) {
            num_clusters = args["k"].as<int>();
        }
        if(args.count("d")) {
            dims = args["d"].as<int>();
        }
        if(args.count("i")) {
            inputFileName = args["i"].as<string>();
        }
        if(args.count("m")) {
            max_num_iter = args["m"].as<int>();
        }
        if(args.count("t")) {
            threshold = args["t"].as<double>();
        }
        if(args.count("c")) {
            cFlag = true;
        }
        if(args.count("s")) {
            seed = args["s"].as<int>();
        }
        if(args.count("g")) {
            gFlag = true;
        }
        if(args.count("f")) {
            fFlag = true;
        }
        if(args.count("p")) {
            pFlag = true;
        }
        if(args.count("a")) {
            answerFileName = args["a"].as<string>();
        }
    } catch (std::exception& e) {
        std::cerr << "Exception 11: " << e.what() << '\n';
    }

    centroidArr.reserve(num_clusters);

    readInputFile();

    srand(seed);    //set random seed

    iter_to_converge = 0;
    if(gFlag) {
        parallelKMeansCUDA();
        printf("%d,%lf\n", iter_to_converge, time_per_iter_in_ms);
    } else if (fFlag) {
        parallelKMeansCUDASharedMem();
        printf("%d,%lf\n", iter_to_converge, time_per_iter_in_ms);
    } else if (pFlag) {
        kMeansPlusPlus();
        printf("%d,%lf\n", iter_to_converge, time_per_iter_in_ms);
    } else {
        sequentialKMeans();
        printf("%d,%lf\n", iter_to_converge, time_per_iter_in_ms);
    }

    //testing
    //compareCentroids();

    if(cFlag) {
        printCentroids();
    } else {
        printPointAssignments();
    }

    // cudaEventRecord(end);
    // cudaEventSynchronize(end);
    // float elapsed = 0;  //in milliseconds
    // cudaEventElapsedTime(&elapsed, start, end);
    // printf("End-To-End Runtime: %lf\n", elapsed);

    // auto endTime = high_resolution_clock::now();
    // duration<double, std::milli> elapsed = endTime - startTime;
    // printf("End-To-End Runtime: %lf\n", elapsed.count());
}