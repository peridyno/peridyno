#include "Algorithm/SimpleMath.h"
#include <glm/glm.hpp>

#include "STLMacro.h"

namespace dyno
{
	template <typename MKey, typename T>
	DYN_FUNC Map<MKey, T>::Map()
	{
	}

	template <typename MKey, typename T>
	DYN_FUNC Pair<MKey, T>* Map<MKey, T>::find(MKey key)
	{
		int ind=leftBound(Pair<MKey,T> (key,T()), m_pairs, m_size);
		return (ind >= m_size || m_pairs[ind] != Pair<MKey, T>(key, T())) ? nullptr : (m_pairs + ind);
	}


	template <typename MKey, typename T>
	DYN_FUNC Pair<MKey, T>* Map<MKey, T>::insert(Pair<MKey, T> pair)
	{
		//return nullptr if the data buffer is full
		if (m_size >= m_maxSize) return nullptr;

		//return the index of the first element that is equel to or biger than val
		int t = leftBound(pair, m_pairs, m_size);

		//if the key is equel, do not insert
		if (m_pairs[t] == pair)
			return m_pairs + t;

		//insert val to t location
		for (int j = m_size; j > t; j--)
		{
			m_pairs[j] = m_pairs[j - 1];
		}
		m_pairs[t] = pair;
		m_size++;

		return m_pairs + t;
	}

	template <typename MKey, typename T>
	DYN_FUNC void Map<MKey, T>::clear()
	{
		m_size = 0;
	}

	template <typename MKey, typename T>
	DYN_FUNC int Map<MKey, T>::size()
	{
		return m_size;
	}

	template <typename MKey, typename T>
	DYN_FUNC bool Map<MKey, T>::empty()
	{
		return m_pairs == nullptr;
	}
}

