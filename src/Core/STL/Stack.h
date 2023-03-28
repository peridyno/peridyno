#ifndef STACK_H
#define STACK_H

#include "Platform.h"
#include "STLBuffer.h"

namespace dyno
{
	/**
	 * @brief Be aware do not use this structure on GPU if the data size is large.
	 * 
	 * @tparam T 
	 */
	template <typename T>
	class Stack : public STLBuffer<T>
	{
	public:
		using iterator = T*;

		DYN_FUNC Stack();

		//DYN_FUNC iterator find(T val);

		DYN_FUNC inline iterator begin() {
			return ::dyno::STLBuffer<T>::m_startLoc;
		};

		DYN_FUNC inline iterator end(){
			return ::dyno::STLBuffer<T>::m_startLoc + m_size;
		}

		DYN_FUNC void clear();

		DYN_FUNC uint size();

		DYN_FUNC T top();
		DYN_FUNC void push(T val);
		DYN_FUNC void pop();
		DYN_FUNC bool empty();

		DYN_FUNC uint count(T val);


	private:
		uint m_size = 0;
	};
}

#include "Stack.inl"

#endif // STACK_H
