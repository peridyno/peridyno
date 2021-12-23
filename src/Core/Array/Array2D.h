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

		void assign(const Array2D<T, DeviceType::GPU>& src);
		void assign(const Array2D<T, DeviceType::CPU>& src);

	private:
		uint m_nx = 0;
		uint m_ny = 0;

		std::vector<T> m_data;
	};

	template<typename T>
	class Array2D<T, DeviceType::GPU>
	{
	public:
		Array2D() {};

		Array2D(uint nx, uint ny)
		{
			this->resize(nx, ny);
		};

		/*!
		*	\brief	Should not release data here, call Release() explicitly.
		*/
		~Array2D() {};

		void resize(uint nx, uint ny);

		void reset();

		void clear();

		inline T* begin() const { return m_data; }

		DYN_FUNC inline uint nx() const { return m_nx; }
		DYN_FUNC inline uint ny() const { return m_ny; }
		DYN_FUNC inline uint pitch() const { return m_pitch; }

		DYN_FUNC inline T operator () (const uint i, const uint j) const
		{
			char* addr = (char*)m_data;
			addr += j * m_pitch;

			return ((T*)addr)[i];
			//return m_data[i + j* m_pitch];
		}

		DYN_FUNC inline T& operator () (const uint i, const uint j)
		{
			char* addr = (char*)m_data;
			addr += j * m_pitch;

			return ((T*)addr)[i];

			//return m_data[i + j* m_pitch];
		}

		DYN_FUNC inline int index(const uint i, const uint j) const
		{
			return i + j * m_nx;
		}

		DYN_FUNC inline T operator [] (const uint id) const
		{
			return m_data[id];
		}

		DYN_FUNC inline T& operator [] (const uint id)
		{
			return m_data[id];
		}

		DYN_FUNC inline uint size() const { return m_nx * m_ny; }
		DYN_FUNC inline bool isCPU() const { return false; }
		DYN_FUNC inline bool isGPU() const { return true; }

		void assign(const Array2D<T, DeviceType::GPU>& src);
		void assign(const Array2D<T, DeviceType::CPU>& src);

	private:
		uint m_nx = 0;
		uint m_ny = 0;
		uint m_pitch = 0;
		T* m_data = nullptr;
	};

	template<typename T>
	using CArray2D = Array2D<T, DeviceType::CPU>;

	template<typename T>
	using DArray2D = Array2D<T, DeviceType::GPU>;
}

#include "Array2D.inl"