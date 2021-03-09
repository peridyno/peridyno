#include "Algorithm/SimpleMath.h"
#include <glm/glm.hpp>

namespace dyno
{
	template <typename T>
	DYN_FUNC Stack<T>::Stack()
	{
	}

	template <typename T>
	DYN_FUNC T Stack<T>::top()
	{
		return m_startLoc[m_size-1];
	}

	template <typename T>
	DYN_FUNC void Stack<T>::push(T val)
	{
		//if the data buffer is full
		//if (m_size >= m_maxSize) 
			//cout<<"Stack is full!"<<endl;

		//if not full add to the end
		m_startLoc[m_size] = val;
		m_size++;
	}

	template <typename T>
	DYN_FUNC void Stack<T>::pop()
	{
		//if the data buffer is empty
		//if (m_size == 0 ) 
			//cout << "Stack is empty!" << endl;

		//else return the data at the top
		m_size--;
	}

	template <typename T>
	DYN_FUNC void Stack<T>::clear()
	{
		m_size = 0;
	}

	template <typename T>
	DYN_FUNC size_t Stack<T>::size()
	{
		return m_size;
	}

	template <typename T>
	DYN_FUNC size_t Stack<T>::count(T val)
	{
		size_t ind = 0;
		size_t num = 0;
		while (ind < m_size)
		{
			if(m_startLoc[ind] == val)
				num++;
			ind++;
		}

		return num;
	}

	template <typename T>
	DYN_FUNC bool Stack<T>::empty()
	{
		return m_startLoc == nullptr;
	}
}

