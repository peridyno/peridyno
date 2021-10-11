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
#include <cassert>
#include <vector>
#include <iostream>
#include <memory>

namespace dyno {

	template<typename T, DeviceType deviceType> class Array;

	template<typename T>
	class Array<T, DeviceType::CPU>
	{
	public:
		Array()
		{
		};

		Array(uint num)
		{
			m_data.resize((size_t)num);
		}

		~Array() {};

		void resize(uint n);

		/*!
		*	\brief	Clear all data to zero.
		*/
		void reset();

		void clear();

		inline const T*	begin() const { return m_data.size() == 0 ? nullptr : &m_data[0]; }
		inline T*	begin() { return m_data.size() == 0 ? nullptr : &m_data[0]; }

		DeviceType	deviceType() { return DeviceType::CPU; }

		inline T& operator [] (unsigned int id)
		{
			return m_data[id];
		}

		inline const T& operator [] (unsigned int id) const
		{
			return m_data[id];
		}

		inline uint size() const { return (uint)m_data.size(); }
		inline bool isCPU() const { return true; }
		inline bool isGPU() const { return false; }
		inline bool isEmpty() const { return m_data.empty(); }

		inline void pushBack(T ele) { m_data.push_back(ele); }

		void assign(const T& val);
		void assign(uint num, const T& val);
		void assign(const Array<T, DeviceType::GPU>& src);
		void assign(const Array<T, DeviceType::CPU>& src);

		friend std::ostream& operator<<(std::ostream &out, const Array<T, DeviceType::CPU>& cArray)
		{
			for (uint i = 0; i < cArray.size(); i++)
			{
				out << i << ": " << cArray[i] << std::endl;
			}

			return out;
		}

	private:
		std::vector<T> m_data;
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
		{
		};

		Array(uint num)
		{
			this->resize(num);
		}

		/*!
		*	\brief	Do not release memory here, call clear() explicitly.
		*/
		~Array() {};

		void resize(const uint n);

		/*!
		*	\brief	Clear all data to zero.
		*/
		void reset();

		/*!
		*	\brief	Free allocated memory.	Should be called before the object is deleted.
		*/
		void clear();

		DYN_FUNC inline const T*	begin() const { return m_data; }
		DYN_FUNC inline T*	begin() { return m_data; }

		DeviceType	deviceType() { return DeviceType::GPU; }

		GPU_FUNC inline T& operator [] (unsigned int id) {
			return m_data[id];
		}

		GPU_FUNC inline T& operator [] (unsigned int id) const {
			return m_data[id];
		}

		DYN_FUNC inline uint size() const { return m_totalNum; }
		DYN_FUNC inline bool isCPU() const { return false; }
		DYN_FUNC inline bool isGPU() const { return true; }
		DYN_FUNC inline bool isEmpty() const { return m_data == nullptr; }

		void assign(const Array<T, DeviceType::GPU>& src);
		void assign(const Array<T, DeviceType::CPU>& src);
		void assign(const std::vector<T>& src);

		void assign(const Array<T, DeviceType::GPU>& src, const uint count, const uint dstOffset = 0, const uint srcOffset = 0);
		void assign(const Array<T, DeviceType::CPU>& src, const uint count, const uint dstOffset = 0, const uint srcOffset = 0);
		void assign(const std::vector<T>& src, const uint count, const uint dstOffset = 0, const uint srcOffset = 0);

		friend std::ostream& operator<<(std::ostream &out, const Array<T, DeviceType::GPU>& dArray)
		{
			Array<T, DeviceType::CPU> hArray;
			hArray.assign(dArray);

			out << hArray;

			return out;
		}

	private:
		T* m_data = 0;
		uint m_totalNum = 0;
	};

	template<typename T>
	using CArray = Array<T, DeviceType::CPU>;

	template<typename T>
	using DArray = Array<T, DeviceType::GPU>;
}

#include "Array.inl"