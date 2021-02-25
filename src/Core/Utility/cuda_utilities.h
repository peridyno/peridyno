#pragma once

#include <assert.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <iostream>
#include "cuda_helper_math.h"

namespace dyno{

#define INVALID -1
#define EPSILON   1e-6
#define M_PI 3.14159265358979323846
#define M_E 2.71828182845904523536

	#define BLOCK_SIZE 64

	using cuint = unsigned int;

	static cuint iDivUp(cuint a, cuint b)
	{
		return (a % b != 0) ? (a / b + 1) : (a / b);
	}

	// compute grid and thread block size for a given number of elements
	static cuint cudaGridSize(cuint totalSize, cuint blockSize)
	{
		int dim = iDivUp(totalSize, blockSize);
		return dim == 0 ? 1 : dim;
	}

	static dim3 cudaGridSize3D(uint3 totalSize, uint blockSize)
	{
		dim3 gridDims;
		gridDims.x = iDivUp(totalSize.x, blockSize);
		gridDims.y = iDivUp(totalSize.y, blockSize);
		gridDims.z = iDivUp(totalSize.z, blockSize);

		gridDims.x = gridDims.x == 0 ? 1 : gridDims.x;
		gridDims.y = gridDims.y == 0 ? 1 : gridDims.y;
		gridDims.z = gridDims.z == 0 ? 1 : gridDims.z;

		return gridDims;
	}

	static dim3 cudaGridSize3D(uint3 totalSize, uint3 blockSize)
	{
		dim3 gridDims;
		gridDims.x = iDivUp(totalSize.x, blockSize.x);
		gridDims.y = iDivUp(totalSize.y, blockSize.y);
		gridDims.z = iDivUp(totalSize.z, blockSize.z);

		gridDims.x = gridDims.x == 0 ? 1 : gridDims.x;
		gridDims.y = gridDims.y == 0 ? 1 : gridDims.y;
		gridDims.z = gridDims.z == 0 ? 1 : gridDims.z;

		return gridDims;
	}

	/** check whether cuda thinks there was an error and fail with msg, if this is the case
	* @ingroup tools
	*/
	static inline void checkCudaError(const char *msg) {
		cudaError_t err = cudaGetLastError();
		if (cudaSuccess != err) {
			//printf( "CUDA error: %d : %s at %s:%d \n", err, cudaGetErrorString(err), __FILE__, __LINE__);
			throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
		}
	}

	// use this macro to make sure no error occurs when cuda functions are called
#ifdef NDEBUG
#define cuSafeCall(X)  X
#else
#define cuSafeCall(X) X; dyno::checkCudaError(#X);
#endif

/**
 * @brief Macro to check cuda errors
 * 
 */
#ifdef NDEBUG
#define cuSynchronize() {}
#else
#define cuSynchronize()	{						\
		char str[200];							\
		cudaDeviceSynchronize();				\
		cudaError_t err = cudaGetLastError();	\
		if (err != cudaSuccess)					\
		{										\
			sprintf(str, "CUDA error: %d : %s at %s:%d \n", err, cudaGetErrorString(err), __FILE__, __LINE__);		\
			throw std::runtime_error(std::string(str));																\
		}																											\
	}
#endif

/**
 * @brief Macro definition for execuation of cuda kernels, note that at lease one block will be executed.
 * 
 * size: indicate how many threads are required in total.
 * Func: kernel function
 */
#define cuExecute(size, Func, ...){						\
		uint pDims = cudaGridSize(size, BLOCK_SIZE);	\
		Func << <pDims, BLOCK_SIZE >> > (				\
		__VA_ARGS__);									\
		cuSynchronize();								\
	}

#define cuExecute3D(size, Func, ...){						\
		uint3 pDims = cudaGridSize3D(size, 8);		\
		dim3 threadsPerBlock(8, 8, 8);		\
		Func << <pDims, threadsPerBlock >> > (				\
		__VA_ARGS__);										\
		cuSynchronize();									\
	}

}// end of namespace dyno
