#pragma once
#include <cassert>
#include <vector>
#include <cuda_runtime.h>
#include <memory>
#include "Platform.h"
#include "MemoryManager.h"

namespace dyno {

	/*!
	*	\class	Array
	*	\brief	This class is designed to be elegant, so it can be directly passed to GPU as parameters.
	*/
	template<typename T, DeviceType deviceType = DeviceType::GPU>
	class Array
	{
	public:
		Array(const std::shared_ptr<MemoryManager<deviceType>> alloc = std::make_shared<DefaultMemoryManager<deviceType>>())
			: m_data(NULL)
			, m_totalNum(0)
			, m_alloc(alloc)
		{
		};

		Array(int num, const std::shared_ptr<MemoryManager<deviceType>> alloc = std::make_shared<DefaultMemoryManager<deviceType>>())
			: m_data(NULL)
			, m_totalNum(num)
			, m_alloc(alloc)
		{
			allocMemory();
		}

		/*!
		*	\brief	Should not release data here, call Release() explicitly.
		*/
		~Array() {};

		void resize(size_t n);

		/*!
		*	\brief	Clear all data to zero.
		*/
		void reset();

		/*!
		*	\brief	Free allocated memory.	Should be called before the object is deleted.
		*/
		void release();

		DYN_FUNC inline T*		begin() { return m_data; }

		DeviceType	getDeviceType() { return deviceType; }

		DYN_FUNC inline T& operator [] (unsigned int id)
		{
			return m_data[id];
		}

		DYN_FUNC inline T operator [] (unsigned int id) const
		{
			return m_data[id];
		}

		DYN_FUNC inline size_t size() { return m_totalNum; }
		DYN_FUNC inline bool isCPU() { return deviceType == DeviceType::CPU; }
		DYN_FUNC inline bool isGPU() { return deviceType == DeviceType::GPU; }
		DYN_FUNC inline bool isEmpty() { return m_data == NULL; }

	protected:
		void allocMemory();
		
	private:
		T* m_data;
		size_t m_totalNum;
		std::shared_ptr<MemoryManager<deviceType>> m_alloc;
	};

	template<typename T>
	using HostArray = Array<T, DeviceType::CPU>;

	template<typename T>
	using DeviceArray = Array<T, DeviceType::GPU>;
}

#include "Array.inl"