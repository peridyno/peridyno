#include "ArrayTools.h"
#include "Algorithm/Scan.h"
#include "Algorithm/Reduction.h"
#include "Array/Array.h"

namespace dyno
{
	template<class ElementType>
	void ArrayMap<ElementType, DeviceType::GPU>::clear()
	{
		m_index.clear();
		m_elements.clear();
		m_maps.clear();
	}

	template<class ElementType>
	bool ArrayMap<ElementType, DeviceType::GPU>::resize(const DArray<int> counts)
	{
		assert(counts.size() > 0);

		if (m_index.size() != counts.size())
		{
			m_index.resize(counts.size());
			m_maps.resize(counts.size());
		}

		m_index.assign(counts);

		Reduction<int> reduce;
		int total_num = reduce.accumulate(m_index.begin(), m_index.size());

		Scan scan;
		scan.exclusive(m_index);

		m_elements.resize(total_num);

		parallel_allocate_for_map<sizeof(ElementType)>(m_maps.begin(), m_elements.begin(), m_elements.size(), m_index);

		return true;
	}


	template<class ElementType>
	bool ArrayMap<ElementType, DeviceType::GPU>::resize(const uint arraySize, const uint eleSize)
	{
		assert(arraySize > 0);
		assert(eleSize > 0);

		if (m_index.size() != arraySize)
		{
			m_index.resize(arraySize);
			m_maps.resize(arraySize);
		}

		CArray<int> hIndex;
		hIndex.resize(arraySize);
		int accNum = 0;
		for (size_t i = 0; i < arraySize; i++)
		{
			hIndex[i] = (int)accNum;
			accNum += eleSize;
		}

		m_index.assign(hIndex);

		m_elements.resize(arraySize*eleSize);

		parallel_allocate_for_map<sizeof(ElementType)>(m_maps.begin(), m_elements.begin(), m_elements.size(), m_index);

		return true;
	}

	template<typename ElementType>
	template<typename ET2>
	bool ArrayMap<ElementType, DeviceType::GPU>::resize(const ArrayMap<ET2, DeviceType::GPU>& src) {
		uint arraySize = src.size();
		if (m_index.size() != arraySize)
		{
			m_index.resize(arraySize);
			m_maps.resize(arraySize);
		}

		m_index.assign(src.index());
		m_elements.resize(src.elementSize());

		parallel_allocate_for_map<sizeof(ElementType)>(m_maps.begin(), m_elements.begin(), m_elements.size(), m_index);

		return true;
	}

	template<class ElementType>
	void ArrayMap<ElementType, DeviceType::GPU>::assign(const ArrayMap<ElementType, DeviceType::GPU>& src)
	{
		m_index.assign(src.index());
		m_elements.assign(src.elements());

		m_maps.assign(src.maps());

		//redirect the element address
		parallel_init_for_map<sizeof(ElementType)>(m_maps.begin(), m_elements.begin(), m_elements.size(), m_index);
	}

	template<class ElementType>
	void ArrayMap<ElementType, DeviceType::GPU>::assign(const ArrayMap<ElementType, DeviceType::CPU>& src)
	{
		m_index.assign(src.index());
		m_elements.assign(src.elements());

		m_maps.assign(src.maps());

		//redirect the element address
		parallel_init_for_map<sizeof(ElementType)>(m_maps.begin(), m_elements.begin(), m_elements.size(), m_index);
	}

	template<class ElementType>
	void ArrayMap<ElementType, DeviceType::GPU>::assign(std::vector<std::map<int,ElementType>>& src)
	{
		size_t indNum = src.size();
		CArray<int> hIndex(indNum);

		CArray<Pair<int,ElementType>> hElements;

		size_t eleNum = 0;
		for (int i = 0; i < src.size(); i++)
		{
			hIndex[i] = (int)eleNum;
			eleNum += src[i].size();

			if (src[i].size() > 0)
			{
				for (typename std::map<int, ElementType>::iterator map_it = src[i].begin(); map_it != src[i].end(); map_it++)
				{
					Pair<int, ElementType> mypair(map_it->first, map_it->second);
					hElements.pushBack(mypair);
				}
			}
		}
		CArray<Map<int, ElementType>> maps;
		maps.resize(indNum);

		m_index.assign(hIndex);
		m_elements.assign(hElements);
		Pair<int, ElementType>* strAdr = m_elements.begin();

		eleNum = 0;
		for (int i = 0; i < src.size(); i++)
		{
			size_t num_i = src[i].size();
			Map<int, ElementType> mmap;
			mmap.assign(strAdr + eleNum, num_i, num_i);
			//printf("DArrayMap.assign(std::vector<std::map<>>): the ptrpair is: %x \n", (strAdr + eleNum));
			maps[i] = mmap;

			eleNum += src[i].size();
		}

		m_maps.assign(maps);
	}

	template<class ElementType>
	bool ArrayMap<ElementType, DeviceType::CPU>::resize(uint num)
	{
		assert(num > 0);

		m_maps.resize(num);
		m_index.resize(num);

		return true;
	}

	template<class ElementType>
	void ArrayMap<ElementType, DeviceType::CPU>::clear()
	{
		for (int i = 0; i < m_maps.size(); i++)
		{
			m_maps[i].clear();
		}

		m_maps.clear();
		m_index.clear();
		m_elements.clear();
	}

	template<class ElementType>
	void ArrayMap<ElementType, DeviceType::CPU>::assign(const ArrayMap<ElementType, DeviceType::CPU>& src)
	{
		m_index.assign(src.index());
		m_elements.assign(src.elements());

		m_maps.assign(src.maps());

		//redirect the element address
		for (int i = 0; i < src.size(); i++)
		{
			m_maps[i].reserve(m_elements.begin() + m_index[i], m_maps[i].size());
		}
	}

	template<class ElementType>
	void ArrayMap<ElementType, DeviceType::CPU>::assign(const ArrayMap<ElementType, DeviceType::GPU>& src)
	{
		m_index.assign(src.index());
		m_elements.assign(src.elements());

		m_maps.assign(src.maps());

		//redirect the element address
		for (int i = 0; i < src.size(); i++)
		{
			m_maps[i].reserve(m_elements.begin() + m_index[i], m_maps[i].size());
		}
	}
}