#include "ArrayTools.h"
#include "Algorithm/Scan.h"
#include "Algorithm/Reduction.h"

namespace dyno
{
	template<class ElementType>
	void ArrayList<ElementType, DeviceType::GPU>::release()
	{
		m_index.clear();
		m_elements.clear();
		m_lists.clear();
	}

	template<class ElementType>
	bool ArrayList<ElementType, DeviceType::GPU>::resize(const GArray<int> counts)
	{
		assert(counts.size() > 0);

		if (m_index.size() != counts.size())
		{
			m_index.resize(counts.size());
			m_lists.resize(counts.size());
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

		parallel_allocate_for_list<sizeof(ElementType)>(m_lists.begin(), m_elements.begin(), m_elements.size(), m_index);

		return true;
	}


	template<class ElementType>
	bool ArrayList<ElementType, DeviceType::GPU>::resize(const size_t arraySize, const size_t eleSize)
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
			hIndex[i] = (int)accNum;
			accNum += eleSize;
		}

		m_index.assign(hIndex);

		m_elements.resize(arraySize*eleSize);

		parallel_allocate_for_list<sizeof(ElementType)>(m_lists.begin(), m_elements.begin(), m_elements.size(), m_index);

		return true;
	}

	template<class ElementType>
	void ArrayList<ElementType, DeviceType::GPU>::assign(const ArrayList<ElementType, DeviceType::GPU>& src)
	{
		m_index.assign(src.index());
		m_elements.assign(src.elements());

		m_lists.assign(src.lists());

		//redirect the element address
		parallel_init_for_list<sizeof(ElementType)>(m_lists.begin(), m_elements.begin(), m_elements.size(), m_index);
	}

	template<class ElementType>
	void ArrayList<ElementType, DeviceType::GPU>::assign(const ArrayList<ElementType, DeviceType::CPU>& src)
	{
		m_index.assign(src.index());
		m_elements.assign(src.elements());

		m_lists.assign(src.lists());

		//redirect the element address
		parallel_init_for_list<sizeof(ElementType)>(m_lists.begin(), m_elements.begin(), m_elements.size(), m_index);
	}

	template<class ElementType>
	void ArrayList<ElementType, DeviceType::GPU>::assign(const std::vector<std::vector<ElementType>>& src)
	{
		size_t indNum = src.size();
		CArray<int> hIndex(indNum);

		CArray<ElementType> hElements;

		size_t eleNum = 0;
		for (int i = 0; i < src.size(); i++)
		{
			hIndex[i] = (int)eleNum;
			eleNum += src[i].size();

			for (int j = 0; j < src[i].size(); j++)
			{
				hElements.pushBack(src[i][j]);
			}
		}

		CArray<List<ElementType>> lists;
		lists.resize(indNum);
			
		m_index.assign(hIndex);
		m_elements.assign(hElements);
		ElementType* stAdr = m_elements.begin();

		eleNum = 0;
		for (int i = 0; i < src.size(); i++)
		{
			size_t num_i = src[i].size();
			List<ElementType> lst;
			lst.assign(stAdr + eleNum, num_i, num_i);
			lists[i] = lst;

			eleNum += src[i].size();
		}

		m_lists.assign(lists);
	}

	template<class ElementType>
	bool ArrayList<ElementType, DeviceType::CPU>::resize(size_t num)
	{
		assert(num > 0);

		m_lists.resize(num);

		return true;
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
	size_t ArrayList<ElementType, DeviceType::CPU>::elementSize()
	{
		size_t totalSize = 0;
		for (int i = 0; i < m_lists.size(); i++)
		{
			totalSize += m_lists[i].size();
		}

		return totalSize;
	}

	template<class ElementType>
	void ArrayList<ElementType, DeviceType::CPU>::assign(const ArrayList<ElementType, DeviceType::CPU>& src)
	{
		m_index.assign(src.index());
		m_elements.assign(src.elements());

		m_lists.assign(src.size());

		//redirect the element address
		for (int i = 0; i < src.size(); i++)
		{
			m_lists[i].reverse(m_elements.begin(), m_lists[i].size());
		}
	}

	template<class ElementType>
	void ArrayList<ElementType, DeviceType::CPU>::assign(const ArrayList<ElementType, DeviceType::GPU>& src)
	{
		m_index.assign(src.index());
		m_elements.assign(src.elements());

		m_lists.assign(src.lists());

		//redirect the element address
		for (int i = 0; i < src.size(); i++)
		{
			m_lists[i].reserve(m_elements.begin(), m_lists[i].size());
		}
	}
}