#ifndef SET_H
#define SET_H

#include "Platform.h"
#include "STLBuffer.h"

namespace dyno
{
	/**
	 * @brief An CPU/GPU implementation of the standard set suitable for small-size data
	 * 
	 * Be aware do not use this structure if the data size is large.
	 * The computation complexity is O(n^2) for some specific situation.
	 * All elements are organized in ascending order
	 * 
	 * @tparam T 
	 */
	template <typename T>
	class Set : public STLBuffer<T>
	{
	public:
		DYN_FUNC Set();
		
		DYN_FUNC T* find(T val);

		DYN_FUNC inline iterator begin() {
			return m_startLoc;
		};

		DYN_FUNC inline iterator end(){
			return m_startLoc + m_size;
		}

		DYN_FUNC void clear();

		DYN_FUNC uint size();
		DYN_FUNC uint count(T val);

		DYN_FUNC T* insert(T val);
		DYN_FUNC bool empty();

		DYN_FUNC int erase(const T val);
		DYN_FUNC void erase(T* val_ptr);


	private:
		uint m_size = 0;
	};



}

#include "Set.inl"

#endif // SET_H
