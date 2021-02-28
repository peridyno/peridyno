#pragma once
#include "Platform.h"
#include <cassert>
#include <vector>
#include <cuda_runtime.h>
#include <memory>
#include "MemoryManager.h"

namespace dyno {

	template<typename T, DeviceType deviceType>
	class Array
	{
	public:
		Array();
		Array(int num);

		~Array() {};

		void resize(size_t n);
		void reset();
		void release();

		DeviceType	deviceType();
	};

	/*!
	*	\class	Array
	*	\brief	This class is designed to be elegant, so it can be directly passed to GPU as parameters.
	*/
	template<typename T>
	class Array<T, DeviceType::GPU>
	{
	public:
		Array()
			: m_data(NULL)
			, m_totalNum(0)
		{
			m_alloc = std::make_shared<DefaultMemoryManager<DeviceType::GPU>>();
		};

		Array(int num)
			: m_data(NULL)
			, m_totalNum(num)
		{
			m_alloc = std::make_shared<DefaultMemoryManager<DeviceType::GPU>>();
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

		DYN_FUNC inline T*	begin() { return m_data; }

		DeviceType	deviceType() { return DeviceType::GPU; }

		GPU_FUNC inline T& operator [] (unsigned int id)
		{
			return m_data[id];
		}

		GPU_FUNC inline T operator [] (unsigned int id) const
		{
			return m_data[id];
		}

		DYN_FUNC inline size_t size() { return m_totalNum; }
		DYN_FUNC inline bool isCPU() { return false; }
		DYN_FUNC inline bool isGPU() { return true; }
		DYN_FUNC inline bool isEmpty() { return m_data == NULL; }

	protected:
		void allocMemory();
		
	private:
		T* m_data;
		size_t m_totalNum;
		std::shared_ptr<MemoryManager<DeviceType::GPU>> m_alloc;
	};


	template<typename T>
	class Array<T, DeviceType::CPU>
	{
	public:
		Array()
		{
		};

		Array(int num)
		{
			m_data.resize(num);
		}

		/*!
		*	\brief	Should not release data here, call Release() explicitly.
		*/
		~Array() { release(); };

		void resize(size_t n);

		/*!
		*	\brief	Clear all data to zero.
		*/
		void reset();

		void release();

		inline T*	begin() { return m_data.size() == 0 ? nullptr : &m_data[0]; }

		DeviceType	deviceType() { return DeviceType::CPU; }

		inline T& operator [] (unsigned int id)
		{
			return m_data[id];
		}

		inline T operator [] (unsigned int id) const
		{
			return m_data[id];
		}

		inline size_t size() { return m_data.size(); }
		inline bool isCPU() { return true; }
		inline bool isGPU() { return false; }
		inline bool isEmpty() { return m_data.empty(); }

		inline void push_back(T ele) { m_data.push_back(ele); }

	private:
		std::vector<T> m_data;
	};


	template<typename T>
	using CArray = Array<T, DeviceType::CPU>;

	template<typename T>
	using GArray = Array<T, DeviceType::GPU>;
}

#include "Array.inl"