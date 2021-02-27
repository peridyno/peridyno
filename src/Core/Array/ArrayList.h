#pragma once
#include "Platform.h"
#include "STL/List.h"

namespace dyno
{
	template< class  ElementType, DeviceType deviceType = DeviceType::GPU>
	class ArrayList
	{
	public:
		ArrayList()
		{
		};

		~ArrayList() {};

		DYN_FUNC inline int size() { return m_lists.size(); }
		
		DYN_FUNC inline List<ElementType>& operator [] (unsigned int id)
		{
			return m_lists[id];
		}

		DYN_FUNC inline List<ElementType> operator [] (unsigned int id) const
		{
			return m_lists[id];
		}

		bool allocate(GArray<int> counts);

		void release()
		{

		}

	private:
		Array<int, deviceType> index;
		Array<ElementType, deviceType> m_elements;
		
		Array<List<ElementType>, deviceType> m_lists;
	};

	template class ArrayList<int>;
}