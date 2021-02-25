#include "MemoryManager.h"

#include <cuda_runtime_api.h>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <assert.h>

#include "Utility.h"

namespace dyno 
{
	template<DeviceType deviceType>
	void DefaultMemoryManager<deviceType>::allocMemory1D(void** ptr, size_t memsize, size_t valueSize)
	{
		switch (deviceType)
		{
		case CPU:
			//assert(*ptr == 0);
			*ptr = malloc(memsize * valueSize);
			//assert(*ptr);
			break;
		case GPU:
			//assert(*ptr == 0);
			cuSafeCall(cudaMalloc(ptr, memsize * valueSize));
			//assert(*ptr);
			break;
		default:
			break;
		}
	}

	template<DeviceType deviceType>
	void DefaultMemoryManager<deviceType>::allocMemory2D(void** ptr, size_t& pitch, size_t height, size_t width, size_t valueSize)
	{
		switch (deviceType)
		{
		case CPU:
			pitch = width * valueSize;
			allocMemory1D(ptr, height * width, valueSize);
			assert(*ptr);
			break;
		case GPU:
			cuSafeCall(cudaMallocPitch(ptr, &pitch, valueSize * width, height));
			assert(*ptr);
			break;
		default:
			break;
		}
	}


	template<DeviceType deviceType>
	void DefaultMemoryManager<deviceType>::initMemory(void* ptr, int value, size_t count)
	{
		switch (deviceType)
		{
		case CPU:
			memset((void*)ptr, value, count);
			break;
		case GPU:
			cudaMemset(ptr, value, count);
			break;
		default:
			break;
		}
	}


	template<DeviceType deviceType>
	void DefaultMemoryManager<deviceType>::releaseMemory(void** ptr)
	{
		switch (deviceType)
		{
		case CPU:
			assert(*ptr != 0);
			free(*ptr);
			*ptr = 0;
			break;
		case GPU:
			assert(*ptr != 0);
			cuSafeCall(cudaFree(*ptr));
			*ptr = 0;
			break;
		default:
			break;
		}
	}

	template<DeviceType deviceType>
	void CudaMemoryManager<deviceType>::allocMemory1D(void** ptr, size_t memsize, size_t valueSize)
	{
		switch (deviceType)
		{
		case CPU:
			assert(*ptr == 0);
			cuSafeCall(cudaMallocHost(ptr, memsize * valueSize));
			assert(*ptr != 0);
			break;
		case GPU:
			DefaultMemoryManager<deviceType>::allocMemory1D(ptr, memsize, valueSize);
			break;
		case UNDEFINED:
			break;
		default:
			break;
		}
	}

	template<DeviceType deviceType>
	void CudaMemoryManager<deviceType>::allocMemory2D(void** ptr, size_t& pitch, size_t height, size_t width, size_t valueSize)
	{
		switch (deviceType)
		{
		case CPU:
			pitch = width * valueSize;
			allocMemory1D(ptr, height * width, valueSize);
			break;
		case GPU:
			cuSafeCall(cudaMallocPitch(ptr, &pitch, valueSize * width, height));
			break;
		case UNDEFINED:
			break;
		default:
			break;
		}
	}

	template<DeviceType deviceType>
	void CudaMemoryManager<deviceType>::initMemory(void* ptr, int value, size_t count)
	{
		DefaultMemoryManager<deviceType>::initMemory(ptr, value, count);
	}


	template<DeviceType deviceType>
	void CudaMemoryManager<deviceType>::releaseMemory(void** ptr)
	{
		switch (deviceType)
		{
		case CPU:
			assert(*ptr != 0);
			cuSafeCall(cudaFreeHost(*ptr));
			*ptr = 0;
			break;
		case GPU:
			DefaultMemoryManager<deviceType>::releaseMemory(ptr);
			break;
		case UNDEFINED:
			break;
		default:
			break;
		}
	}
}