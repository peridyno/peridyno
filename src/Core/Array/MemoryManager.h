#pragma once
#ifdef _MSC_VER
#pragma warning(disable: 4661) // disable warning 4345
#endif

#include <map>
#include <string>
#include <cuda_runtime.h>
#include "Platform.h"

namespace dyno 
{
	template<DeviceType deviceType>
	class MemoryManager {

	public:
		MemoryManager() {};

		virtual ~MemoryManager() {
		}

		virtual void allocMemory1D(void** ptr, size_t memsize, size_t valueSize) = 0;

		virtual void allocMemory2D(void** ptr, size_t& pitch, size_t height, size_t width, size_t valueSize) = 0;

		virtual void initMemory(void* ptr, int value, size_t count) = 0;

		virtual void releaseMemory(void** ptr) = 0;
	};

	/**
	 * Allocator allows allocation, deallocation and copying depending on memory_space_type
	 *
	 * \ingroup tools
	 */
	template<DeviceType deviceType>
	class DefaultMemoryManager : public MemoryManager<deviceType> {

	public:
		DefaultMemoryManager() {};

		virtual ~DefaultMemoryManager() {
		}

		void allocMemory1D(void** ptr, size_t memsize, size_t valueSize) override;

		void allocMemory2D(void** ptr, size_t& pitch, size_t height, size_t width, size_t valueSize) override;

		void initMemory(void* ptr, int value, size_t count) override;

		void releaseMemory(void** ptr) override;

	};


	/**
	 * @brief allocator that uses cudaMallocHost for allocations in host_memory_space
	 */
	template<DeviceType deviceType>
	class CudaMemoryManager : public DefaultMemoryManager<deviceType> {

	public:
		CudaMemoryManager() {};

		virtual ~CudaMemoryManager() {
		}

		void allocMemory1D(void** ptr, size_t memsize, size_t valueSize) override;

		void allocMemory2D(void** ptr, size_t& pitch, size_t height, size_t width, size_t valueSize) override;

		void initMemory(void* ptr, int value, size_t count) override;

		void releaseMemory(void** ptr) override;

	};


	template class DefaultMemoryManager<DeviceType::CPU>;
	template class DefaultMemoryManager<DeviceType::GPU>;
	template class CudaMemoryManager<DeviceType::CPU>;
	template class CudaMemoryManager<DeviceType::GPU>;
}