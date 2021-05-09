#include "ArrayTools.h"
#include "Algorithm/Scan.h"
#include "Algorithm/Reduction.h"
//#include "Array/Array.h"

namespace dyno
{
	template<class ElementType>
	void ArrayMap<ElementType, DeviceType::GPU>::release()
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

		if (total_num <= 0)
		{
			return false;
		}

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
	void ArrayMap<ElementType, DeviceType::GPU>::assign(const std::vector<Map<int,ElementType>>& src)
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
				for (int j = 0; j < src[i].size(); j++)
				{
					hElements.pushBack(src[i].m_pairs[j]);
				}
			}
		}
		m_index.assign(hIndex);
		m_elements.assign(hElements);

		m_maps.assign(src);
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