#ifndef LIST_H
#define LIST_H

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
	class List : public STLBuffer<T>
	{
	public:
		using iterator = T*;

		DYN_FUNC List();
		
		DYN_FUNC iterator find(T val);

		DYN_FUNC inline iterator begin() {
			return m_startLoc;
		};

		DYN_FUNC inline iterator end(){
			return m_startLoc + m_size;
		}

		DYN_FUNC void clear();

		DYN_FUNC int size();

		DYN_FUNC inline iterator insert(T val);
		DYN_FUNC inline iterator atomicInsert(T val);

		DYN_FUNC inline T front();
		DYN_FUNC inline T back();

		DYN_FUNC bool empty();

	private:
		int m_size = 0;
	};

}

#include "List.inl"

#endif // LIST_H
