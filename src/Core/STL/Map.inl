#include "Utility/SimpleMath.h"
#include <glm/glm.hpp>

namespace dyno
{
	template <typename MKey, typename T>
	DYN_FUNC Map<MKey, T>::Map()
	{
	}

	template <typename MKey, typename T>
	DYN_FUNC Pair<MKey, T>* Map<MKey, T>::find(MKey key)
	{
		return leftBound(key, data, m_size);
	}


	template <typename MKey, typename T>
	DYN_FUNC Pair<MKey, T>* Map<MKey, T>::insert(Pair<MKey, T> pair)
	{
		int t = leftBound(val, data, m_size);

		if (t == INVALID) return nullptr;
		if (data[t] == val) return data + t;

		for (int j = m_size; j > t; t--)
		{
			data[j] = data[j - 1];
		}
		data[t] = val;

		return data + t;
	}

	template <typename MKey, typename T>
	DYN_FUNC void Map<T>::clear()
	{
		m_size = 0;
	}

	template <typename MKey, typename T>
	DYN_FUNC int Map<T>::size()
	{
		return m_size;
	}

	template <typename MKey, typename T>
	DYN_FUNC bool Map<T>::empty()
	{
		return data == nullptr;
	}
}

