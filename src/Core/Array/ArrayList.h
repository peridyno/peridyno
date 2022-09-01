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

		inline uint size() const { return m_lists.size(); }
		uint elementSize();

		inline List<ElementType>& operator [] (unsigned int id)
		{
			return m_lists[id];
		}

		inline const List<ElementType>& operator [] (unsigned int id) const
		{
			return m_lists[id];
		}

		inline bool isCPU() const { return true; }
		inline bool isGPU() const { return false; }
		inline bool isEmpty() const { return m_lists.empty(); }

		void clear();

		void assign(const ArrayList<ElementType, DeviceType::CPU>& src);
		void assign(const ArrayList<ElementType, DeviceType::GPU>& src);

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

		const CArray<int>& index() const { return m_index; }
		const CArray<ElementType>& elements() const { return m_elements; }
		const CArray<List<ElementType>>& lists() const { return m_lists; }

		/*!
		 *	\brief	To avoid erroneous shallow copy.
		 */
		ArrayList<ElementType, DeviceType::CPU>& operator=(const ArrayList<ElementType, DeviceType::CPU> &) = delete;

	private:
		CArray<int> m_index;
		CArray<ElementType> m_elements;

		CArray<List<ElementType>> m_lists;
	};

	template<class ElementType>
	class ArrayList<ElementType, DeviceType::GPU>
	{
	public:
		ArrayList()
		{
		};

		/*!
		*	\brief	Do not release memory here, call clear() explicitly.
		*/
		~ArrayList() {};

		/**
		 * @brief Pre-allocate GPU space for
		 *
		 * @param counts
		 * @return true
		 * @return false
		 */
		bool resize(const DArray<int>& counts);
		bool resize(const uint arraySize, const uint eleSize);


		bool resize(uint num);

		template<typename ET2>
		bool resize(const ArrayList<ET2, DeviceType::GPU>& src);

		DYN_FUNC inline uint size() const { return m_lists.size(); }
		DYN_FUNC inline uint elementSize() const { return m_elements.size(); }

		GPU_FUNC inline List<ElementType>& operator [] (unsigned int id) {
			return m_lists[id];
		}

		GPU_FUNC inline List<ElementType>& operator [] (unsigned int id) const {
			return m_lists[id];
		}

		DYN_FUNC inline bool isCPU() const { return false; }
		DYN_FUNC inline bool isGPU() const { return true; }
		DYN_FUNC inline bool isEmpty() const { return m_index.size() == 0; }

		void clear();

		void assign(const ArrayList<ElementType, DeviceType::GPU>& src);
		void assign(const ArrayList<ElementType, DeviceType::CPU>& src);
		void assign(const std::vector<std::vector<ElementType>>& src);

		friend std::ostream& operator<<(std::ostream &out, const ArrayList<ElementType, DeviceType::GPU>& aList)
		{
			ArrayList<ElementType, DeviceType::CPU> hList;
			hList.assign(aList);
			out << hList;

			return out;
		}

		const DArray<int>& index() const { return m_index; }
		const DArray<ElementType>& elements() const { return m_elements; }
		const DArray<List<ElementType>>& lists() const { return m_lists; }

		/*!
		*	\brief	To avoid erroneous shallow copy.
		*/
		ArrayList<ElementType, DeviceType::GPU>& operator=(const ArrayList<ElementType, DeviceType::GPU> &) = delete;

	private:
		DArray<int> m_index;
		DArray<ElementType> m_elements;

		DArray<List<ElementType>> m_lists;
	};

	template<typename T>
	using DArrayList = ArrayList<T, DeviceType::GPU>;

	template<typename T>
	using CArrayList = ArrayList<T, DeviceType::CPU>;
}

#include "ArrayList.inl"
