/*************************************************************************************************
 * File: sobelFilter.cu
 * Date: 09/27/2017
 *
 * Compiling: Requires a Nvidia CUDA capable graphics card and the Nvidia GPU Computing Toolkit.
 *            Linux: nvcc -Wno-deprecated-gpu-targets -O3 -o edge sobelFilter.cu lodepng.cpp -Xcompiler -fopenmp
 *
 * Usage:   Linux: >> edge [filename.png]
 *
 * Description: This file is meant to handle all the sobel filter functions as well as the main
 *      function. Each sobel filter function runs in a different way than the others, one is a basic
 *      sobel filter running through just the cpu on a single thread, another runs through openmp
 *      to parallelize the single thread cpu function, and the last one runs through a NVIDIA gpu
 *      to parallelize the function onto the many cores available on the gpu.
 *************************************************************************************************/

#include <thread>
#include <chrono>
#include <time.h>
#include <iostream>
#include <math.h>
#include "imageLoader.cpp"

#define GRIDVAL 20.0

void conv_cpu3x3(const byte* orig, byte* cpu, const int* __restrict__ maskX, const unsigned int width, const unsigned int height);
void conv_cpu7x7(const byte* orig, byte* cpu, const int* __restrict__ maskX, const unsigned int width, const unsigned int height);
void conv_cpu13x13(const byte* orig, byte* cpu, const int* __restrict__ maskX, const unsigned int width, const unsigned int height);
void conv_cpu17x17(const byte* orig, byte* cpu, const int* __restrict__ maskX, const unsigned int width, const unsigned int height);

__global__ void conv_gpu3x3(const byte* orig, byte* cpu, const int* __restrict__ maskX, const unsigned int width, const unsigned int height) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    float dx=0.0;

    int cont=0;

    if( x > 0 && y > 0 && x < width-1 && y < height-1) {
    	//Mascara 3x3
		for(int yy=-1; yy<=1;yy++){
			for(int xx=-1; xx<=1;xx++){
				dx += (maskX[cont]* orig[(y-yy)*width + (x-xx)]);
				cont+=1;
			}
		}
		cpu[y*width + x] = sqrt((dx*dx)+(dx*dx));
    }
}

__global__ void conv_gpu7x7(const byte* orig, byte* cpu, const int* __restrict__ maskX, const unsigned int width, const unsigned int height) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    float dx=0.0;

    int cont=0;

    if( x > 2 && y > 2 && x < width-1 && y < height-1) {
    	//Mascara 7x7
		for(int yy=-3; yy<=3;yy++){
			for(int xx=-3; xx<=3;xx++){
				dx += (maskX[cont]* orig[(y-yy)*width + (x-xx)]);
				cont+=1;
			}
		}
		cpu[y*width + x] = sqrt((dx*dx)+(dx*dx));
    }
}

__global__ void conv_gpu13x13(const byte* orig, byte* cpu, const int* __restrict__ maskX, const unsigned int width, const unsigned int height) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    float dx=0.0;

    int cont=0;

    if( x > 5 && y > 5 && x < width-1 && y < height-1) {
    	//Mascara 7x7
		for(int yy=-6; yy<=6;yy++){
			for(int xx=-6; xx<=6;xx++){
				dx += (maskX[cont]* orig[(y-yy)*width + (x-xx)]);
				cont+=1;
			}
		}
		cpu[y*width + x] = sqrt( (dx*dx)+(dx*dx));
    }
}

__global__ void conv_gpu17x17(const byte* orig, byte* cpu, const int* __restrict__ maskX, const unsigned int width, const unsigned int height) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    float dx=0.0;

    int cont=0;

    if( x > 7 && y > 7 && x < width-1 && y < height-1) {
    	//Mascara 7x7
		for(int yy=-8; yy<=8;yy++){
			for(int xx=-8; xx<=8;xx++){
				dx += (maskX[cont]* orig[(y-yy)*width + (x-xx)]);
				cont+=1;
			}
		}
		cpu[y*width + x] = sqrt((dx*dx)+(dx*dx));
    }
}

int main(int argc, char*argv[]) {
    /** Check command line arguments **/
    if(argc != 2) {
        printf("%s: Invalid number of command line arguments. Exiting program\n", argv[0]);
        printf("Usage: %s [image.png]", argv[0]);
        return 1;
    }
    /** Gather CUDA device properties **/
	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, 0);
	int cores = devProp.multiProcessorCount;
	switch (devProp.major)
	{
	case 2: // Fermi
		if (devProp.minor == 1) cores *= 48;
		else cores *= 32; break;
	case 3: // Kepler
		cores *= 192; break;
	case 5: // Maxwell
		cores *= 128; break;
	case 6: // Pascal
		if (devProp.minor == 1) cores *= 128;
		else if (devProp.minor == 0) cores *= 64;
		break;
    }

    time_t rawTime;time(&rawTime);
    struct tm* curTime = localtime(&rawTime);
    char timeBuffer[80] = "";
    strftime(timeBuffer, 80, "\nPráctica Convolucion (%c)\n", curTime);
    printf("%s", timeBuffer);

    imgData origImg = loadImage(argv[1]);
    imgData cpuImg(new byte[origImg.width*origImg.height], origImg.width, origImg.height);
    imgData gpuImg(new byte[origImg.width*origImg.height], origImg.width, origImg.height);

    int* deviceMaskX;

    int mask_size, mask_side;

    mask_side = 17;
    mask_size = mask_side * mask_side * sizeof(float);

    //Mascara 3x3
    /*
    int MaskdX[9] = {2, 1, 2,
			   	   	 0, 0, 0,
			        -2,-1,-2};

	*/

    //Mascara 7x7
    /*
    int MaskdX[49] = {-1,-1,-1,-1,-1,-1,-1,
		  	  	  	  -1,-2,-2,-2,-2,-2,-1,
		  	  	  	  -1,-2,-3,-3,-3,-2,-1,
		  	  	  	   0, 0, 0, 0, 0, 0, 0,
		  	           1, 2, 3, 3, 3, 2, 1,
		  	           1, 2, 2, 2, 2, 2, 1,
		  	           1, 1, 1, 1, 1, 1, 1};
	*/

    //Mascara 13x13
    /*
    int MaskdX[169] = {7, 6, 5, 4, 3, 2,  1, 2, 3, 4, 5, 6, 7,
    		  	  	   7, 6, 5, 4, 3, 2,  1, 2, 3, 4, 5, 6, 7,
    		  	  	   7, 6, 5, 4, 3, 2,  1, 2, 3, 4, 5, 6, 7,
         		  	   7, 6, 5, 4, 3, 2,  1, 2, 3, 4, 5, 6, 7,
    		  	       7, 6, 5, 4, 3, 2,  1, 2, 3, 4, 5, 6, 7,
    		  	       7, 6, 5, 4, 3, 2,  1, 2, 3, 4, 5, 6, 7,
    		  	       0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0,
    		  	      -7,-6,-5,-4,-3,-2, -1,-2,-3,-4,-5,-6,-7,
    		  	      -7,-6,-5,-4,-3,-2, -1,-2,-3,-4,-5,-6,-7,
    		  	      -7,-6,-5,-4,-3,-2, -1,-2,-3,-4,-5,-6,-7,
    		  	      -7,-6,-5,-4,-3,-2, -1,-2,-3,-4,-5,-6,-7,
    		  	      -7,-6,-5,-4,-3,-2, -1,-2,-3,-4,-5,-6,-7,
    		  	      -7,-6,-5,-4,-3,-2, -1,-2,-3,-4,-5,-6,-7};
	*/

    /*
    //Mascara 13x13
    int MaskdX[289] = {9, 8, 7, 6, 5, 4, 3, 2,  1, 2, 3, 4, 5, 6, 7, 8, 9,
    				   9, 8, 7, 6, 5, 4, 3, 2,  1, 2, 3, 4, 5, 6, 7, 8, 9,
    				   9, 8, 7, 6, 5, 4, 3, 2,  1, 2, 3, 4, 5, 6, 7, 8, 9,
    				   9, 8, 7, 6, 5, 4, 3, 2,  1, 2, 3, 4, 5, 6, 7, 8, 9,
    				   9, 8, 7, 6, 5, 4, 3, 2,  1, 2, 3, 4, 5, 6, 7, 8, 9,
    				   9, 8, 7, 6, 5, 4, 3, 2,  1, 2, 3, 4, 5, 6, 7, 8, 9,
    				   9, 8, 7, 6, 5, 4, 3, 2,  1, 2, 3, 4, 5, 6, 7, 8, 9,
    				   9, 8, 7, 6, 5, 4, 3, 2,  1, 2, 3, 4, 5, 6, 7, 8, 9,
    				   0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0
    				  -9,-8,-7,-6,-5,-4,-3,-2, -1,-2,-3,-4,-5,-6,-7,-8,-9,
    				  -9,-8,-7,-6,-5,-4,-3,-2, -1,-2,-3,-4,-5,-6,-7,-8,-9,
    				  -9,-8,-7,-6,-5,-4,-3,-2, -1,-2,-3,-4,-5,-6,-7,-8,-9,
    				  -9,-8,-7,-6,-5,-4,-3,-2, -1,-2,-3,-4,-5,-6,-7,-8,-9,
    				  -9,-8,-7,-6,-5,-4,-3,-2, -1,-2,-3,-4,-5,-6,-7,-8,-9,
    				  -9,-8,-7,-6,-5,-4,-3,-2, -1,-2,-3,-4,-5,-6,-7,-8,-9,
    				  -9,-8,-7,-6,-5,-4,-3,-2, -1,-2,-3,-4,-5,-6,-7,-8,-9,
    				  -9,-8,-7,-6,-5,-4,-3,-2, -1,-2,-3,-4,-5,-6,-7,-8,-9};
	*/

    memset(cpuImg.pixels, 0, (origImg.width*origImg.height));

    auto c = std::chrono::system_clock::now();
    conv_cpu17x17(origImg.pixels, cpuImg.pixels, MaskdX, origImg.width, origImg.height);
    std::chrono::duration<double> time_cpu = std::chrono::system_clock::now() - c;

    byte *gpu_orig, *gpuInput;

    cudaMalloc( (void**)&gpu_orig, (origImg.width * origImg.height));
    cudaMalloc( (void**)&gpuInput, (origImg.width * origImg.height));
    cudaMalloc( (void**)&deviceMaskX, mask_size);


    cudaMemcpy(gpu_orig, origImg.pixels, (origImg.width*origImg.height), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuInput,0, (origImg.width*origImg.height), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMaskX, MaskdX, mask_size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(GRIDVAL, GRIDVAL, 1);
    dim3 numBlocks(ceil(origImg.width/GRIDVAL), ceil(origImg.height/GRIDVAL), 1);

    c = std::chrono::system_clock::now();
    conv_gpu17x17<<<numBlocks, threadsPerBlock>>>(gpu_orig, gpuInput, deviceMaskX, origImg.width, origImg.height);
    cudaError_t cudaerror = cudaDeviceSynchronize(); // waits for completion, returns error code
    std::chrono::duration<double> time_gpu = std::chrono::system_clock::now() - c;

    cudaMemcpy(gpuImg.pixels, gpuInput, (origImg.width*origImg.height), cudaMemcpyDeviceToHost);

    printf("\nEntrada %s: %d filas x %d columnas\n", argv[1], origImg.height, origImg.width);
    printf("CPU %dx%d tiempo de ejecucion    = %*.1f msec\n", mask_side, mask_side, 5, 1000*time_cpu.count());
    printf("GPU %dx%d tiempo de ejecucion   = %*.3f msec\n", mask_side, mask_side, 5, 1000*time_gpu.count());
    printf("\nGPU %dx%d -> CPU aceleracion:%*.1fX \n", mask_side, mask_side, 12, (1000*time_cpu.count())/(1000*time_gpu.count()));
    printf("\n");


    writeImage(argv[1], "gpu17x17", gpuImg);
    writeImage(argv[1], "cpu17x17", cpuImg);

    cudaFree(gpu_orig); cudaFree(gpuInput);
    return 0;
}

void conv_cpu3x3(const byte* orig, byte* cpu, const int* __restrict__ maskX, const unsigned int width, const unsigned int height) {
	int cont=0;
	int dx=0;
    for(int y = 1; y < height-1; y++) {
        for(int x = 1; x < width-1; x++) {
        	for(int yy=-1;yy<=1;yy++){
        		for(int xx=-1; xx<=1;xx++){
        			dx += (maskX[cont]* orig[(y-yy)*width + (x-xx)]);
        			cont+=1;
        		}
        	}
        	cont=0;
        	cpu[y*width + x] = sqrt((dx*dx)+(dx*dx));
        	dx=0.0;
        }
    }
}

void conv_cpu7x7(const byte* orig, byte* cpu, const int* __restrict__ maskX, const unsigned int width, const unsigned int height) {
	int cont=0;
	int dx=0;
    for(int y = 3; y < height-1; y++) {
        for(int x = 3; x < width-1; x++) {
        	for(int yy=-3;yy<=3;yy++){
        		for(int xx=-3; xx<=3;xx++){
        			dx += (maskX[cont]* orig[(y-yy)*width + (x-xx)]);
        			cont+=1;
        		}
        	}
        	cont=0;
        	cpu[y*width + x] = sqrt((dx*dx)+(dx*dx));
        	dx=0.0;
        }
    }
}

void conv_cpu13x13(const byte* orig, byte* cpu, const int* __restrict__ maskX, const unsigned int width, const unsigned int height) {
	int cont=0;
	int dx=0;
    for(int y = 6; y < height-1; y++) {
        for(int x = 6; x < width-1; x++) {
        	for(int yy=-6;yy<=6;yy++){
        		for(int xx=-6; xx<=6;xx++){
        			dx += (maskX[cont]* orig[(y-yy)*width + (x-xx)]);
        			cont+=1;
        		}
        	}
        	cont=0;
        	cpu[y*width + x] = sqrt((dx*dx)+(dx*dx));
        	dx=0.0;
        }
    }
}

void conv_cpu17x17(const byte* orig, byte* cpu, const int* __restrict__ maskX, const unsigned int width, const unsigned int height) {
	int cont=0;
	int dx=0;
    for(int y = 8; y < height-1; y++) {
        for(int x = 8; x < width-1; x++) {
        	for(int yy=-8;yy<=8;yy++){
        		for(int xx=-8; xx<=8;xx++){
        			dx += (maskX[cont]* orig[(y-yy)*width + (x-xx)]);
        			cont+=1;
        		}
        	}
        	cont=0;
        	cpu[y*width + x] = sqrt((dx*dx)+(dx*dx));
        	dx=0.0;
        }
    }
}

