#include "Utility.h"
#include "Allocator.h"

namespace dyno
{
	template<class ElementType>
	void ArrayList<ElementType, DeviceType::GPU>::release()
	{
		m_index.release();
		m_elements.release();
		m_lists.release();
	}

	template<class ElementType>
	bool ArrayList<ElementType, DeviceType::GPU>::resize(GArray<int> counts)
	{
		assert(counts.size() > 0);

		if (m_index.size() != counts.size())
		{
			m_index.resize(counts.size());
			m_lists.resize(counts.size());
		}

		Function1Pt::copy(m_index, counts);

		Reduction<int> reduce;
		int total_num = reduce.accumulate(m_index.begin(), m_index.size());

		if (total_num <= 0)
		{
			return false;
		}

		Scan scan;
		scan.exclusive(m_index);

		m_elements.resize(total_num);

		parallel_allocate_for_list<sizeof(ElementType)>(m_lists.begin(), m_elements.begin(), m_elements.size(), m_index);

		return true;
	}


	template<class ElementType>
	bool ArrayList<ElementType, DeviceType::GPU>::resize(size_t arraySize, size_t eleSize)
	{
		assert(arraySize > 0);
		assert(eleSize > 0);

		if (m_index.size() != arraySize)
		{
			m_index.resize(arraySize);
			m_lists.resize(arraySize);
		}

		CArray<int> hIndex;
		hIndex.resize(arraySize);
		int accNum = 0;
		for (size_t i = 0; i < arraySize; i++)
		{
			hIndex[i] = accNum;
			accNum += eleSize;
		}

		Function1Pt::copy(m_index, hIndex);

		m_elements.resize(arraySize*eleSize);

		parallel_allocate_for_list<sizeof(ElementType)>(m_lists.begin(), m_elements.begin(), m_elements.size(), m_index);

		return true;
	}

	template<class ElementType>
	bool ArrayList<ElementType, DeviceType::CPU>::resize(size_t num)
	{
		assert(num > 0);

		m_lists.resize(counts.size());
	}

	template<class ElementType>
	void ArrayList<ElementType, DeviceType::CPU>::clear()
	{
		for (int i = 0; i < m_lists.size(); i++)
		{
			m_lists[i].clear();
		}

		m_lists.clear();
	}

	template<class ElementType>
	void ArrayList<ElementType, DeviceType::CPU>::pushBack(std::list<ElementType>& lst)
	{
		m_lists.push_back(lst);
	}

	template<class ElementType>
	size_t ArrayList<ElementType, DeviceType::CPU>::elementSize()
	{
		size_t totalSize = 0;
		for (int i = 0; i < m_lists.size(); i++)
		{
			totalSize += m_lists[i].size();
		}

		return totalSize;
	}
}