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
#include <list>
#include "STL/List.h"

namespace dyno
{
	template<class ElementType, DeviceType deviceType>
	class ArrayList
	{
	public:
		ArrayList()
		{
		};

		~ArrayList() {};
	};

	template<class ElementType>
	class ArrayList<ElementType, DeviceType::GPU>
	{
	public:
		ArrayList()
		{
		};

		~ArrayList() {};

		/**
		 * @brief Pre-allocate GPU space for
		 * 
		 * @param counts 
		 * @return true 
		 * @return false 
		 */
		bool resize(GArray<int> counts);
		bool resize(size_t arraySize, size_t eleSize);

		DYN_FUNC inline int size() { return m_lists.size(); }
		DYN_FUNC inline int elementSize() { return m_elements.size(); }
		
		GPU_FUNC inline List<ElementType>& operator [] (unsigned int id)
		{
			return m_lists[id];
		}

		GPU_FUNC inline List<ElementType> operator [] (unsigned int id) const
		{
			return m_lists[id];
		}

		void release();

	private:
		GArray<int> m_index;
		GArray<ElementType> m_elements;
		
		GArray<List<ElementType>> m_lists;
	};


	template<class ElementType>
	class ArrayList<ElementType, DeviceType::CPU>
	{
	public:
		ArrayList() {};
		~ArrayList() {};

		bool resize(size_t num);

		inline size_t size() { return m_lists.size(); }
		size_t elementSize();

		inline std::list<ElementType>& operator [] (unsigned int id)
		{
			return m_lists[id];
		}

		inline std::list<ElementType> operator [] (unsigned int id) const
		{
			return m_lists[id];
		}

		void clear();

		void pushBack(std::list<ElementType>& lst);

	private:
		std::vector<std::list<ElementType>> m_lists;
	};


	template<typename T>
	using GArrayList = ArrayList<T, DeviceType::GPU>;

	template<typename T>
	using CArrayList = ArrayList<T, DeviceType::CPU>;
}

#include "ArrayList.inl"
