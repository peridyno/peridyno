#include "Math/SimpleMath.h"
#include <glm/glm.hpp>

#include "STLMacro.h"

namespace dyno
{
	template <typename T>
	DYN_FUNC MultiSet<T>::MultiSet()
		: STLBuffer()
	{
	}

	template <typename T>
	DYN_FUNC T* MultiSet<T>::find(T val)
	{
		int ind = leftBound(val, m_startLoc, m_size);

		return ind >= m_size || m_startLoc[ind] != val ? nullptr : m_startLoc + ind;
	}

	template <typename T>
	DYN_FUNC T* MultiSet<T>::insert(T val)
	{
		//return nullptr if the data buffer is full
		if (m_size >= m_maxSize) return nullptr;

		//return the index of the last element that is equal to or smaller than val
		int ind = rightBound(val, m_startLoc, m_size);

		int ind_plus = ind + 1;

		//move all element backward
		for (int j = m_size; j > ind_plus; j--)
		{
			m_startLoc[j] = m_startLoc[j - 1];
		}

		//insert val to ind-th location.
		m_startLoc[ind_plus] = val;
		m_size++;

		return m_startLoc + ind_plus;
	}

	template <typename T>
	DYN_FUNC void MultiSet<T>::clear()
	{
		m_size = 0;
	}

	template <typename T>
	DYN_FUNC uint MultiSet<T>::size()
	{
		return m_size;
	}

	template <typename T>
	DYN_FUNC uint MultiSet<T>::count(T val)
	{
		int ind = leftBound(val, m_startLoc, m_size);
		if (ind >= m_size) return 0;

		size_t num = 0;
		while (m_startLoc[ind] == val && ind < m_size)
		{
			num++;
			ind++;
		}

		return num;
	}

	template <typename T>
	DYN_FUNC bool MultiSet<T>::empty()
	{
		return m_startLoc == nullptr;
	}
}

