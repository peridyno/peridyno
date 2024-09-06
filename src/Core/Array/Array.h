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
#include <cmath>

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
			mData.resize((size_t)num);
		}

		~Array() {};

		void resize(uint n);

		/*!
		*	\brief	Clear all data to zero.
		*/
		void reset();

		void clear();

		inline const T*	begin() const { return mData.size() == 0 ? nullptr : &mData[0]; }
		inline T*	begin() { return mData.size() == 0 ? nullptr : &mData[0]; }

		inline const std::vector<T>* handle() const { return &mData; }
		inline std::vector<T>* handle() { return &mData; }

		DeviceType	deviceType() { return DeviceType::CPU; }

		inline T& operator [] (unsigned int id)
		{
			return mData[id];
		}

		inline const T& operator [] (unsigned int id) const
		{
			return mData[id];
		}

		inline uint size() const { return (uint)mData.size(); }
		inline bool isCPU() const { return true; }
		inline bool isGPU() const { return false; }
		inline bool isEmpty() const { return mData.empty(); }

		inline void pushBack(T ele) { mData.push_back(ele); }

		void assign(const T& val);
		void assign(uint num, const T& val);

	#ifndef NO_BACKEND
		void assign(const Array<T, DeviceType::GPU>& src);
	#endif

		void assign(const Array<T, DeviceType::CPU>& src);
		void assign(const std::vector<T>& src);

		friend std::ostream& operator<<(std::ostream &out, const Array<T, DeviceType::CPU>& cArray)
		{
			for (uint i = 0; i < cArray.size(); i++)
			{
				out << i << ": " << cArray[i] << std::endl;
			}

			return out;
		}

	private:
		std::vector<T> mData;
	};

	template<typename T>
	void Array<T, DeviceType::CPU>::resize(const uint n)
	{
		mData.resize(n);
	}

	template<typename T>
	void Array<T, DeviceType::CPU>::clear()
	{
		mData.clear();
	}

	template<typename T>
	void Array<T, DeviceType::CPU>::reset()
	{
		memset((void*)mData.data(), 0, mData.size()*sizeof(T));
	}

	template<typename T>
	void Array<T, DeviceType::CPU>::assign(const Array<T, DeviceType::CPU>& src)
	{
		if (mData.size() != src.size())
			this->resize(src.size());

		memcpy(this->begin(), src.begin(), src.size() * sizeof(T));
	}

	template<typename T>
	void Array<T, DeviceType::CPU>::assign(const std::vector<T>& src)
	{
		if (mData.size() != src.size())
			this->resize(src.size());

		mData.assign(src.begin(), src.end());
	}

	template<typename T>
	void Array<T, DeviceType::CPU>::assign(const T& val)
	{
		mData.assign(mData.size(), val);
	}

	template<typename T>
	void Array<T, DeviceType::CPU>::assign(uint num, const T& val)
	{
		mData.assign(num, val);
	}

	template<typename T>
	using CArray = Array<T, DeviceType::CPU>;
}

#ifdef CUDA_BACKEND
	#include "Backend/Cuda/Array/Array.inl"
#endif

#ifdef VK_BACKEND
	#include "Backend/Vulkan/Array/Array.inl"
#endif