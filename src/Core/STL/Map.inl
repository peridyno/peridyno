#include "Math/SimpleMath.h"
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
	DYN_FUNC Pair<MKey, T>* Map<MKey, T>::plusInsert(Pair<MKey, T> pair)
	{
		//return the index of the first element that is equel to or biger than val
		int t = leftBound(pair, m_pairs, m_size);

		//if the key is equel, add values together
		if (m_pairs[t] == pair)
		{
			m_pairs[t].second += pair.second;
			return m_pairs + t;
		}

		//return nullptr if the data buffer is full
		if (m_size >= m_maxSize) return nullptr;

		//insert val to t location
		for (int j = m_size; j > t; j--)
		{
			m_pairs[j] = m_pairs[j - 1];
		}
		m_pairs[t] = pair;
		m_size++;

		return m_pairs + t;
	}

#ifdef CUDA_BACKEND
	template <typename MKey, typename T>
	DYN_FUNC Pair<MKey, T>* Map<MKey, T>::atomicInsert(Pair<MKey, T> pair)
	{
		//return nullptr if the data buffer is full
		if (m_size >= this->m_maxSize) return nullptr;

		const uint one = 1;
		int index = atomicAdd(&(this->m_size), one);

		this->m_pairs[index] = pair;

		return this->m_pairs + index;
	}
#endif

	template <typename MKey, typename T>
	DYN_FUNC void Map<MKey, T>::clear()
	{
		m_size = 0;
	}

	template <typename MKey, typename T>
	DYN_FUNC uint Map<MKey, T>::size()
	{
		return m_size;
	}

	template <typename MKey, typename T>
	DYN_FUNC uint Map<MKey, T>::maxSize()
	{
		return m_maxSize;
	}

	template <typename MKey, typename T>
	DYN_FUNC bool Map<MKey, T>::empty()
	{
		return m_pairs == nullptr;
	}
}

