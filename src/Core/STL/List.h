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

		DYN_FUNC inline T& operator [] (uint id) {
			return m_startLoc[id];
		}

		DYN_FUNC inline T& operator [] (uint id) const {
			return m_startLoc[id];
		}

		DYN_FUNC void assign(T* beg, int num, int buffer_size) {
			m_startLoc = beg;
			m_size = num;
			m_maxSize = buffer_size;
		}

		DYN_FUNC void clear();

		DYN_FUNC uint size();

		DYN_FUNC inline iterator insert(T val);
		GPU_FUNC inline iterator atomicInsert(T val);

		DYN_FUNC inline T front();
		DYN_FUNC inline T back();

		DYN_FUNC bool empty();

	private:
		uint m_size = 0;
	};

}

#include "List.inl"

#endif // LIST_H
