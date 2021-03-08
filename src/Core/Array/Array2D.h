/**
 * Copyright 2021 Xiaowei He
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#include <assert.h>
#include <cuda_runtime.h>
#include "Platform.h"
#include "MemoryManager.h"
namespace dyno {

#define INVALID -1

	template<typename T, DeviceType deviceType = DeviceType::GPU>
	class Array2D
	{
	public:
		Array2D(const std::shared_ptr<MemoryManager<deviceType>> alloc = std::make_shared<DefaultMemoryManager<deviceType>>())
			: m_nx(0)
			, m_ny(0)
			, m_pitch(0)
			, m_totalNum(0)
			, m_data(NULL)
			, m_alloc(alloc)
		{};

		Array2D(int nx, int ny, const std::shared_ptr<MemoryManager<deviceType>> alloc = std::make_shared<DefaultMemoryManager<deviceType>>())
			: m_nx(nx)
			, m_ny(ny)
			, m_pitch(0)
			, m_totalNum(nx*ny)
			, m_data(NULL)
			, m_alloc(alloc)
		{
			AllocMemory();
		};

		/*!
		*	\brief	Should not release data here, call Release() explicitly.
		*/
		~Array2D() { };

		void resize(size_t nx, size_t ny);

		void reset();

		void clear();

		inline T* data() const { return m_data; }

		DYN_FUNC inline size_t nx() const { return m_nx; }
		DYN_FUNC inline size_t ny() const { return m_ny; }

		DYN_FUNC inline T operator () (const size_t i, const size_t j) const
		{
			return m_data[i + j* m_pitch];
		}

		DYN_FUNC inline T& operator () (const size_t i, const size_t j)
		{
			return m_data[i + j* m_pitch];
		}

		DYN_FUNC inline int index(const size_t i, const size_t j) const
		{
			return i + j * m_pitch;
		}

		DYN_FUNC inline T operator [] (const size_t id) const
		{
			return m_data[id];
		}

		DYN_FUNC inline T& operator [] (const size_t id)
		{
			return m_data[id];
		}

		DYN_FUNC inline size_t size() const { return m_totalNum; }
		DYN_FUNC inline bool isCPU() const { return deviceType; }
		DYN_FUNC inline bool isGPU() const { return deviceType; }

	public:
		void AllocMemory();

	private:
		size_t m_nx;
		size_t m_ny;
		size_t m_pitch;
		size_t m_totalNum;
		T*	m_data;
		std::shared_ptr<MemoryManager<deviceType>> m_alloc;
	};

	template<typename T, DeviceType deviceType>
	void Array2D<T, deviceType>::resize(size_t nx, size_t ny)
	{
		if (NULL != m_data) clear();
		m_nx = nx;	m_ny = ny;	m_totalNum = m_nx*m_ny;
		AllocMemory();
	}

	template<typename T, DeviceType deviceType>
	void Array2D<T, deviceType>::reset()
	{
		m_alloc->initMemory((void*)m_data, 0, m_pitch * m_ny * sizeof(T));
	}

	template<typename T, DeviceType deviceType>
	void Array2D<T, deviceType>::clear()
	{
		if (m_data != NULL)
		{
			m_alloc->releaseMemory((void**)&m_data);
		}

		m_data = NULL;
		m_nx = 0;
		m_ny = 0;
		m_pitch = 0;
		m_totalNum = 0;
	}

	template<typename T, DeviceType deviceType>
	void Array2D<T, deviceType>::AllocMemory()
	{
		m_alloc->allocMemory2D((void**)&m_data, m_pitch, m_nx, m_ny, sizeof(T));
		m_pitch /= sizeof(T);

		reset();
	}

	template<typename T>
	using CArray2D = Array2D<T, DeviceType::CPU>;

	template<typename T>
	using GArray2D = Array2D<T, DeviceType::GPU>;
}
