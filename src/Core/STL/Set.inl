#include "STLMacro.h"
#include "Algorithm/SimpleMath.h"
#include <glm/glm.hpp>

namespace dyno
{
	template <typename T>
	DYN_FUNC Set<T>::Set()
		: STLBuffer<T>()
	{
	}

	template <typename T>
	DYN_FUNC T* Set<T>::find(T val)
	{
		int ind = leftBound(val, m_startLoc, m_size);

		return ind >= m_size || m_startLoc[ind] != val ? nullptr : m_startLoc + ind;
	}


	template <typename T>
	DYN_FUNC T* Set<T>::insert(T val)
	{
		//return nullptr if the data buffer is full
		if (m_size >= m_maxSize) return nullptr;

		//return the index of the first element that is equal to or greater than val
		int ind = leftBound(val, m_startLoc, m_size);

		if (ind == m_size)
		{
			m_startLoc[ind] = val;
			m_size++;

			return m_startLoc + ind;
		};

		//return the original address if val is found
		if (m_startLoc[ind] == val) 
			return m_startLoc + ind;
		else
		{
			//if found, move all element backward
			for (size_t j = m_size; j > ind; j--)
			{
				m_startLoc[j] = m_startLoc[j - 1];
			}
			
			//insert val into location ind.
			m_startLoc[ind] = val;
			m_size++;

			return m_startLoc + ind;
		}
	}

	template <typename T>
	DYN_FUNC void Set<T>::clear()
	{
		m_size = 0;
	}

	template <typename T>
	DYN_FUNC size_t Set<T>::size()
	{
		return m_size;
	}

	template <typename T>
	DYN_FUNC size_t Set<T>::count(T val)
	{
		return find(val) ? 1 : 0;
	}

	template <typename T>
	DYN_FUNC bool Set<T>::empty()
	{
		return m_startLoc == nullptr;
	}
}

