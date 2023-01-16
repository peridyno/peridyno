/**
 * Copyright 2021 Lixin Ren
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
#include <map>
#include <vector>
#include <iostream>
#include "STL/Map.h"
#include "Array/Array.h"

namespace dyno {
	template<class ElementType, DeviceType deviceType> class ArrayMap;

	template<class ElementType>
	class ArrayMap<ElementType, DeviceType::CPU>
	{
	public:
		ArrayMap() {};
		~ArrayMap() {}

		void clear();

		bool resize(uint num);

		inline uint size() const { return m_maps.size(); }
		inline uint elementSize() const { return m_elements.size(); }

		inline Map<int,ElementType>& operator [] (unsigned int id)
		{
			return m_maps[id];
		}

		inline const Map<int,ElementType>& operator [] (unsigned int id) const
		{
			return m_maps[id];
		}

		inline bool isCPU() const { return true; }
		inline bool isGPU() const { return false; }
		inline bool isEmpty() const { return m_maps.empty(); }



		void assign(const ArrayMap<ElementType, DeviceType::CPU>& src);
		void assign(const ArrayMap<ElementType, DeviceType::GPU>& src);

		friend std::ostream& operator<<(std::ostream &out, const ArrayMap<ElementType, DeviceType::CPU>& aMap)
		{
			out << std::endl;
			for (int i = 0; i < aMap.size(); i++)
			{
				Map<int,ElementType> mmap = aMap[i];

				out << "Map " << i << " (" << mmap.size() << "):";
				for (auto it = mmap.begin(); it != mmap.end(); it++)
				{
					std::cout << "  key:" << it->first << " value:" << it->second;
				}
				out << std::endl;
			}
			return out;
		}

		const CArray<uint>& index() const { return m_index; }
		const CArray<Pair<int,ElementType>>& elements() const { return m_elements; }
		const CArray<Map<int,ElementType>>& maps() const { return m_maps; }

		/*!
		 *	\brief	To avoid erroneous shallow copy.
		 */
		ArrayMap<ElementType, DeviceType::CPU>& operator=(const ArrayMap<ElementType, DeviceType::CPU> &) = delete;

	private:
		CArray<uint> m_index;
		CArray<Pair<int,ElementType>> m_elements;

		CArray<Map<int,ElementType>> m_maps;
	};

	template<class ElementType>
	class ArrayMap<ElementType, DeviceType::GPU>
	{
	public:
		ArrayMap()
		{
		};

		/*!
		*	\brief	Do not release memory here, call clear() explicitly.
		*/
		~ArrayMap() {};

		/**
		 * @brief Pre-allocate GPU space for
		 *
		 * @param counts
		 * @return true
		 * @return false
		 */
		bool resize(const DArray<uint> counts);
		bool resize(const uint arraySize, const uint eleSize);

		template<typename ET2>
		bool resize(const ArrayMap<ET2, DeviceType::GPU>& src);

		DYN_FUNC inline uint size() const { return m_maps.size(); }
		DYN_FUNC inline uint elementSize() const { return m_elements.size(); }

		GPU_FUNC inline Map<int,ElementType>& operator [] (unsigned int id) {
			return m_maps[id];
		}

		GPU_FUNC inline const Map<int,ElementType>& operator [] (unsigned int id) const {
			return m_maps[id];
		}

		DYN_FUNC inline bool isCPU() const { return false; }
		DYN_FUNC inline bool isGPU() const { return true; }
		DYN_FUNC inline bool isEmpty() const { return m_index.size() == 0; }

		void clear();

		void assign(const ArrayMap<ElementType, DeviceType::GPU>& src);
		void assign(const ArrayMap<ElementType, DeviceType::CPU>& src);
		void assign(std::vector<std::map<int, ElementType>>& src);

		friend std::ostream& operator<<(std::ostream &out, const ArrayMap<ElementType, DeviceType::GPU>& aMap)
		{
			ArrayMap<ElementType, DeviceType::CPU> hMap;
			hMap.assign(aMap);
			out << hMap;

			return out;
		}

		const DArray<uint>& index() const { return m_index; }
		const DArray<Pair<int,ElementType>>& elements() const { return m_elements; }
		const DArray<Map<int,ElementType>>& maps() const { return m_maps; }

		/*!
		*	\brief	To avoid erroneous shallow copy.
		*/
		ArrayMap<ElementType, DeviceType::GPU>& operator=(const ArrayMap<ElementType, DeviceType::GPU> &) = delete;

	private:
		DArray<uint> m_index;
		DArray<Pair<int,ElementType>> m_elements;

		DArray<Map<int,ElementType>> m_maps;
	};

	template<typename T>
	using DArrayMap = ArrayMap<T, DeviceType::GPU>;

	template<typename T>
	using CArrayMap = ArrayMap<T, DeviceType::CPU>;
}

#include "ArrayMap.inl"
