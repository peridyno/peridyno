#pragma once
#include "Platform.h"
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

		bool resize(GArray<int> counts);

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

	template<typename T>
	using GArrayList = ArrayList<T, DeviceType::GPU>;
}

#include "ArrayList.inl"
