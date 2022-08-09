#include "Algorithm/SimpleMath.h"
#include <glm/glm.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
namespace dyno
{
	template <typename T>
	DYN_FUNC List<T>::List()
	{
	}

	template <typename T>
	DYN_FUNC T* List<T>::find(T val)
	{
		return nullptr;
	}


	template <typename T>
	DYN_FUNC T* List<T>::insert(T val)
	{
		//return nullptr if the data buffer is full
		if (m_size >= this->m_maxSize) return nullptr;

		this->m_startLoc[m_size] = val;
		m_size++;

		return this->m_startLoc + m_size - 1;;
	}

	template <typename T>
	GPU_FUNC T* List<T>::atomicInsert(T val)
	{
		//return nullptr if the data buffer is full
		if (m_size >= this->m_maxSize) return nullptr;

		//int index = atomicAdd(&m_size, 1);
		//int index = atomicAdd(&size, 1);
		int index = 0;
		this->m_startLoc[index] = val;

		return this->m_startLoc + index;
	}



	template <typename T>
	DYN_FUNC void List<T>::clear()
	{
		m_size = 0;
	}

	template <typename T>
	DYN_FUNC uint List<T>::size()
	{
		return m_size;
	}

	template <typename T>
	DYN_FUNC bool List<T>::empty()
	{
		return this->m_startLoc == nullptr;
	}

	template <typename T>
	DYN_FUNC T List<T>::front()
	{
		return this->m_startLoc[0];
	}

	template <typename T>
	DYN_FUNC T List<T>::back()
	{
		return this->m_startLoc[m_size - 1];
	}
}

