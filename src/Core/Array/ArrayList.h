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
#include <vector>
#include <iostream>

#include "Platform.h"

#include "STL/List.h"
#include "Array/Array.h"

namespace dyno {
	template<class ElementType, DeviceType deviceType> class ArrayList;

	template<class ElementType>
	class ArrayList<ElementType, DeviceType::CPU>
	{
	public:
		ArrayList() {};
		~ArrayList() {}

		bool resize(uint num);

		inline uint size() const { return mLists.size(); }
		uint elementSize();

		uint size(uint id)
		{
			return id == mIndex.size() - 1 ? mElements.size() - mIndex[id] : mIndex[id + 1] - mIndex[id];
		}

		inline List<ElementType>& operator [] (unsigned int id)
		{
			return mLists[id];
		}

		inline const List<ElementType>& operator [] (unsigned int id) const
		{
			return mLists[id];
		}

		inline bool isCPU() const { return true; }
		inline bool isGPU() const { return false; }
		inline bool isEmpty() const { return mLists.empty(); }

		void clear();

		void assign(const ArrayList<ElementType, DeviceType::CPU>& src);

	#ifndef NO_BACKEND
		void assign(const ArrayList<ElementType, DeviceType::GPU>& src);
	#endif

		friend std::ostream& operator<<(std::ostream &out, const ArrayList<ElementType, DeviceType::CPU>& aList)
		{
			out << std::endl;
			for (uint i = 0; i < aList.size(); i++)
			{
				List<ElementType> lst = aList[i];
				out << "List " << i << " (" << lst.size() << "):";
				for (auto it = lst.begin(); it != lst.end(); it++)
				{
					std::cout << " " << *it;
				}
				out << std::endl;
			}
			return out;
		}

		const CArray<uint>& index() const { return mIndex; }
		const CArray<ElementType>& elements() const { return mElements; }
		const CArray<List<ElementType>>& lists() const { return mLists; }

		/*!
		 *	\brief	To avoid erroneous shallow copy.
		 */
		ArrayList<ElementType, DeviceType::CPU>& operator=(const ArrayList<ElementType, DeviceType::CPU> &) = delete;

	private:
		CArray<uint> mIndex;
		CArray<ElementType> mElements;

		CArray<List<ElementType>> mLists;
	};

	template<class ElementType>
	void ArrayList<ElementType, DeviceType::CPU>::clear()
	{
		for (int i = 0; i < mLists.size(); i++)
		{
			mLists[i].clear();
		}

		mLists.clear();
		mIndex.clear();
		mElements.clear();
	}

	template<class ElementType>
	uint ArrayList<ElementType, DeviceType::CPU>::elementSize()
	{
		return mElements.size();
	}

	template<class ElementType>
	void ArrayList<ElementType, DeviceType::CPU>::assign(const ArrayList<ElementType, DeviceType::CPU>& src)
	{
		mIndex.assign(src.index());
		mElements.assign(src.elements());

		mLists.assign(src.lists());

		//redirect the element address
		for (uint i = 0; i < src.size(); i++)
		{
			mLists[i].reserve(mElements.begin() + mIndex[i], mLists[i].size());
		}
	}

	template<typename T>
	using CArrayList = ArrayList<T, DeviceType::CPU>;
}

#ifdef CUDA_BACKEND
	#include "Backend/Cuda/Array/ArrayList.inl"
#endif

#ifdef VK_BACKEND
	#include "Backend/Vulkan/Array/ArrayList.inl"
#endif
