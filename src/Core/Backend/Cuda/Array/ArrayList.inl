#include "ArrayTools.h"
#include "Algorithm/Scan.h"
#include "Algorithm/Reduction.h"

namespace dyno
{
	template<class ElementType>
	void ArrayList<ElementType, DeviceType::GPU>::clear()
	{
		m_index.clear();
		m_elements.clear();
		m_lists.clear();
	}

	template<class ElementType>
	bool ArrayList<ElementType, DeviceType::GPU>::resize(const DArray<int>& counts)
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

		Scan scan;
		scan.exclusive(m_index);

		//printf("total num 2 = %d\n", total_num);

		m_elements.resize(total_num);
		
		parallel_allocate_for_list<sizeof(ElementType)>(m_lists.begin(), m_elements.begin(), m_elements.size(), m_index);

		return true;
	}


	template<class ElementType>
	bool ArrayList<ElementType, DeviceType::GPU>::resize(const uint arraySize, const uint eleSize)
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

	template<typename ElementType>
	template<typename ET2>
	bool ArrayList<ElementType, DeviceType::GPU>::resize(const ArrayList<ET2, DeviceType::GPU>& src) {
		uint arraySize = src.size();
		if (m_index.size() != arraySize)
		{
			m_index.resize(arraySize);
			m_lists.resize(arraySize);
		}

		m_index.assign(src.index());
		m_elements.resize(src.elementSize());

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
	bool ArrayList<ElementType, DeviceType::CPU>::resize(uint num)
	{
		assert(num > 0);

		m_index.resize(num);
		m_lists.resize(num);
		
		return true;
	}

	template<class ElementType>
	bool ArrayList<ElementType, DeviceType::GPU>::resize(uint num)
	{
		assert(num > 0);

		m_index.resize(num);
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
		m_index.clear();
		m_elements.clear();
	}

	template<class ElementType>
	uint ArrayList<ElementType, DeviceType::CPU>::elementSize()
	{
		return m_elements.size();
	}

	template<class ElementType>
	void ArrayList<ElementType, DeviceType::CPU>::assign(const ArrayList<ElementType, DeviceType::CPU>& src)
	{
		m_index.assign(src.index());
		m_elements.assign(src.elements());

		m_lists.assign(src.lists());

		//redirect the element address
		for (uint i = 0; i < src.size(); i++)
		{
			m_lists[i].reserve(m_elements.begin() + m_index[i], m_lists[i].size());
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
			m_lists[i].reserve(m_elements.begin() + m_index[i], m_lists[i].size());
		}
	}
}