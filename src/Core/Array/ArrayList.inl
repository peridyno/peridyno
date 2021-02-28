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


}