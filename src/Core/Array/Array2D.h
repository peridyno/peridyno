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
#include "Platform.h"

namespace dyno {
	template<typename T, DeviceType deviceType> class Array2D;

	template<typename T>
	class Array2D<T, DeviceType::CPU>
	{
	public:
		Array2D() {};

		Array2D(uint nx, uint ny) {
			this->resize(nx, ny);
		};

		/*!
		*	\brief	Should not release data here, call Release() explicitly.
		*/
		~Array2D() {
			this->clear();
		};

		void resize(uint nx, uint ny);

		void reset();

		void clear();

		inline const std::vector<T>* handle() const { return &m_data; }
		inline std::vector<T>* handle() { return &m_data; }

		inline const T* begin() const { return m_data.data(); }

		inline uint nx() const { return m_nx; }
		inline uint ny() const { return m_ny; }

		inline T operator () (const uint i, const uint j) const
		{
			return m_data[i + j * m_nx];
		}

		inline T& operator () (const uint i, const uint j)
		{
			return m_data[i + j * m_nx];
		}

		inline int index(const uint i, const uint j) const
		{
			return i + j * m_nx;
		}

		inline T operator [] (const uint id) const
		{
			return m_data[id];
		}

		inline T& operator [] (const uint id)
		{
			return m_data[id];
		}

		inline uint size() const { return (uint)m_data.size(); }
		inline bool isCPU() const { return false; }
		inline bool isGPU() const { return true; }

#ifndef NO_BACKEND
		void assign(const Array2D<T, DeviceType::GPU>& src);
#endif

		void assign(const Array2D<T, DeviceType::CPU>& src);

	private:
		uint m_nx = 0;
		uint m_ny = 0;

		std::vector<T> m_data;
	};

	template<typename T>
	void Array2D<T, DeviceType::CPU>::resize(uint nx, uint ny)
	{
		if (m_data.size() != 0) clear();

		m_data.resize(nx * ny);
		m_nx = nx;
		m_ny = ny;
	}

	template<typename T>
	void Array2D<T, DeviceType::CPU>::reset()
	{
		std::fill(m_data.begin(), m_data.end(), 0);
	}

	template<typename T>
	void dyno::Array2D<T, DeviceType::CPU>::clear()
	{
		m_data.clear();

		m_nx = 0;
		m_ny = 0;
	}

	template<typename T>
	void Array2D<T, DeviceType::CPU>::assign(const Array2D<T, DeviceType::CPU>& src)
	{
		if (m_nx != src.size() || m_ny != src.size()) {
			this->resize(src.nx(), src.ny());
		}

		m_data.assign(src.m_data);
	}

	template<typename T>
	using CArray2D = Array2D<T, DeviceType::CPU>;
}

#ifdef CUDA_BACKEND
#include "Backend/Cuda/Array/Array2D.inl"
#endif

#ifdef VK_BACKEND
#include "Backend/Vulkan/Array/Array2D.inl"
#endif
