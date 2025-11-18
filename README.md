# Exp5 Bubble Sort and Merge sort in CUDA
# Name: Vignesh M
# Reference No: 212223240176
# Date:13-11-2025
**Objective:**
Implement Bubble Sort and Merge Sort on the GPU using CUDA, analyze the efficiency of this sorting algorithm when parallelized, and explore the limitations of Bubble Sort and Merge Sort for large datasets.
## AIM:
Implement Bubble Sort and Merge Sort on the GPU using CUDA to enhance the performance of sorting tasks by parallelizing comparisons and swaps within the sorting algorithm.

Code Overview:
You will work with the provided CUDA implementation of Bubble Sort and Merge Sort. The code initializes an unsorted array, applies the Bubble Sort, Merge Sort algorithm in parallel on the GPU, and returns the sorted array as output.

## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC, Google Colab with NVCC Compiler, CUDA Toolkit installed, and sample datasets for testing.

## PROCEDURE:

Tasks:

a. Modify the Kernel:

Implement Bubble Sort and Merge Sort using CUDA by assigning each comparison and swap task to individual threads.
Ensure the kernel checks boundaries to avoid out-of-bounds access, particularly for edge cases.
b. Performance Analysis:

Measure the execution time of the CUDA Bubble Sort with different array sizes (e.g., 512, 1024, 2048 elements).
Experiment with various block sizes (e.g., 16, 32, 64 threads per block) to analyze their effect on execution time and efficiency.
c. Comparison:

Compare the performance of the CUDA-based Bubble Sort and Merge Sort with a CPU-based Bubble Sort and Merge Sort implementation.
Discuss the differences in execution time and explain the limitations of Bubble Sort and Merge Sort when parallelized on the GPU.
## PROGRAM:
```python
!pip install git+https://github.com/andreinechaev/nvcc4jupyter.git
%load_ext nvcc4jupyter
%%cuda
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <chrono>
#include <algorithm> // for min()

// ================= CPU FUNCTIONS ===================

// Bubble Sort CPU
void bubbleSortCPU(int arr[], int n) {
    for (int i = 0; i < n - 1; i++)
        for (int j = 0; j < n - i - 1; j++)
            if (arr[j] > arr[j + 1])
                std::swap(arr[j], arr[j + 1]);
}

// Merge function CPU
void mergeHost(int *arr, int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;
    int *L = (int*)malloc(n1*sizeof(int));
    int *R = (int*)malloc(n2*sizeof(int));

    for (int i=0;i<n1;i++) L[i] = arr[left+i];
    for (int i=0;i<n2;i++) R[i] = arr[mid+1+i];

    int i=0, j=0, k=left;
    while(i<n1 && j<n2)
        arr[k++] = (L[i] <= R[j]) ? L[i++] : R[j++];
    while(i<n1) arr[k++] = L[i++];
    while(j<n2) arr[k++] = R[j++];

    free(L); free(R);
}

// Merge Sort CPU
void mergeSortCPU(int *arr, int n) {
    for (int size=1; size<n; size*=2)
        for (int left=0; left+size<n; left+=2*size) {
            int mid = left+size-1;
            int right = std::min(left+2*size-1, n-1);
            mergeHost(arr, left, mid, right);
        }
}

// ================= GPU KERNELS ===================

// Bubble Sort Kernel (single block)
__global__ void bubbleSortKernel(int *d_arr, int n) {
    int idx = threadIdx.x;
    for(int i=0; i<n-1; i++){
        if(idx < n-1-i){
            if(d_arr[idx] > d_arr[idx+1]){
                int temp = d_arr[idx];
                d_arr[idx] = d_arr[idx+1];
                d_arr[idx+1] = temp;
            }
        }
        __syncthreads();
    }
}

// Merge function device
__device__ void mergeDevice(int *arr, int left, int mid, int right, int *temp){
    int i=left, j=mid+1, k=left;
    while(i<=mid && j<=right) temp[k++] = (arr[i]<=arr[j]) ? arr[i++] : arr[j++];
    while(i<=mid) temp[k++] = arr[i++];
    while(j<=right) temp[k++] = arr[j++];
    for(i=left;i<=right;i++) arr[i] = temp[i];
}

// Merge Sort Kernel
__global__ void mergeSortKernel(int *d_arr, int *d_temp, int n){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for(int size=1; size<n; size*=2){
        int left = 2*size*tid;
        if(left < n){
            int mid = min(left+size-1, n-1);
            int right = min(left+2*size-1, n-1);
            mergeDevice(d_arr, left, mid, right, d_temp);
        }
        __syncthreads();
    }
}

// ================= GPU FUNCTIONS ===================

void bubbleSortGPU(int *arr, int n){
    int *d_arr;
    cudaMalloc(&d_arr, n*sizeof(int));
    cudaMemcpy(d_arr, arr, n*sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    bubbleSortKernel<<<1, n>>>(d_arr, n);  // single block
    cudaDeviceSynchronize();

    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float ms=0; cudaEventElapsedTime(&ms,start,stop);

    cudaMemcpy(arr, d_arr, n*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_arr);

    printf("Bubble Sort GPU took %f ms\n", ms);
}

void mergeSortGPU(int *arr, int n){
    int *d_arr, *d_temp;
    cudaMalloc(&d_arr, n*sizeof(int));
    cudaMalloc(&d_temp, n*sizeof(int));
    cudaMemcpy(d_arr, arr, n*sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    int block = 128;
    int grid = (n + block -1)/block;
    mergeSortKernel<<<grid, block>>>(d_arr, d_temp, n);
    cudaDeviceSynchronize();

    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float ms=0; cudaEventElapsedTime(&ms,start,stop);

    cudaMemcpy(arr, d_arr, n*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_arr); cudaFree(d_temp);

    printf("Merge Sort GPU took %f ms\n", ms);
}

// ================= MAIN ===================

int main(){
    int sizes[] = {500, 1000}; // Array sizes to test

    for(int s=0;s<2;s++){
        int n = sizes[s];
        int *arr = (int*)malloc(n*sizeof(int));

        printf("\nArray Size: %d\n", n);

        // Generate random array
        for(int i=0;i<n;i++) arr[i] = rand()%1000;

        auto start = std::chrono::high_resolution_clock::now();
        bubbleSortCPU(arr, n);
        auto end = std::chrono::high_resolution_clock::now();
        printf("Bubble Sort CPU took %f ms\n",
               std::chrono::duration<double,std::milli>(end-start).count());

        for(int i=0;i<n;i++) arr[i] = rand()%1000;
        bubbleSortGPU(arr, n);

        for(int i=0;i<n;i++) arr[i] = rand()%1000;
        start = std::chrono::high_resolution_clock::now();
        mergeSortCPU(arr, n);
        end = std::chrono::high_resolution_clock::now();
        printf("Merge Sort CPU took %f ms\n",
               std::chrono::duration<double,std::milli>(end-start).count());

        for(int i=0;i<n;i++) arr[i] = rand()%1000;
        mergeSortGPU(arr, n);

        free(arr);
    }

    return 0;
}
```

## OUTPUT:
<img width="375" height="268" alt="image" src="https://github.com/user-attachments/assets/ced23a99-16c1-4b92-aef1-8954e17d90f8" />


## RESULT:
Thus, the program has been executed using CUDA to implement Bubble Sort and Merge Sort on the GPU using CUDA and analyze the efficiency of this sorting algorithm when parallelized.
