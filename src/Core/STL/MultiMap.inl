#include "Math/SimpleMath.h"
#include <glm/glm.hpp>

#include "STLMacro.h"

namespace dyno
{
	template <typename MKey, typename T>
	DYN_FUNC MultiMap<MKey, T>::MultiMap()
	{
	}

	template <typename MKey, typename T>
	DYN_FUNC Pair<MKey, T>* MultiMap<MKey, T>::find(MKey key)
	{
		int ind = leftBound(Pair<MKey, T>(key, T()), m_startLoc, m_size);

		return ind >= m_size || m_startLoc[ind] != Pair<MKey, T>(key, T()) ? nullptr : m_startLoc + ind;
	}

	template <typename MKey, typename T>
	DYN_FUNC Pair<MKey, T>* MultiMap<MKey, T>::insert(Pair<MKey, T> pair)
	{
		//return nullptr if the data buffer is full
		if (m_size >= m_maxSize) return nullptr;

		//return the index of the last element that is equal to or smaller than val
		int ind = rightBound(pair, m_startLoc, m_size);

		int ind_plus = ind + 1;

		//move all element backward
		for (size_t j = m_size; j > ind_plus; j--)
		{
			m_startLoc[j] = m_startLoc[j - 1];
		}

		//insert val to ind-th location.
		m_startLoc[ind_plus] = pair;
		m_size++;

		return m_startLoc + ind_plus;
	}

	template <typename MKey, typename T>
	DYN_FUNC T& MultiMap<MKey, T>::operator[](MKey key)
	{
		Pair<MKey, T>* ret = find(key);

		if (ret == nullptr)
			ret = insert(Pair<MKey, T>(key, T()));

		return ret->second;
	}


	template <typename MKey, typename T>
	DYN_FUNC const T& MultiMap<MKey, T>::operator[](MKey key) const
	{
		Pair<MKey, T>* ret = find(key);

		if (ret == nullptr)
			ret = insert(Pair<MKey, T>(key, T()));

		return ret->second;
	}


	template <typename MKey, typename T>
	DYN_FUNC void MultiMap<MKey, T>::clear()
	{
		m_size = 0;
	}

	template <typename MKey, typename T>
	DYN_FUNC uint MultiMap<MKey, T>::size()
	{
		return m_size;
	}

	template <typename MKey, typename T>
	DYN_FUNC uint MultiMap<MKey, T>::count(MKey key)
	{
		int ind = leftBound(Pair<MKey, T>(key, T(0)), m_startLoc, m_size);
		if (ind >= m_size) return 0;

		size_t num = 0;
		while (m_startLoc[ind] == val && ind < m_size)
		{
			num++;
			ind++;
		}

		return num;
	}

	template <typename MKey, typename T>
	DYN_FUNC bool MultiMap<MKey, T>::empty()
	{
		return m_startLoc == nullptr;
	}
}

