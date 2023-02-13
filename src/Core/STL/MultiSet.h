#ifndef MultiSet_H
#define MultiSet_H

#include "Platform.h"
#include "STLBuffer.h"

namespace dyno
{
	/**
	 * @brief An CPU/GPU implementation of the standard multiset suitable for small-size data
	 * 
	 * Be aware do not use this structure if the data size is large,
	 * because the computation complexity is O(n^2) for some specific situation.
	 * 
	 * All elements are organized in non-descending order.
	 * 
	 * @tparam T 
	 */
	template <typename T>
	class MultiSet : public STLBuffer<T>
	{
	public:
		using iterator = T*;

		DYN_FUNC MultiSet();
		
		DYN_FUNC iterator find(T val);

		DYN_FUNC inline iterator begin() {
			return m_startLoc;
		};

		DYN_FUNC inline iterator end(){
			return m_startLoc + m_size;
		}

		DYN_FUNC void clear();

		DYN_FUNC uint size();
		DYN_FUNC uint count(T val);

		DYN_FUNC iterator insert(T val);
		DYN_FUNC bool empty();

		DYN_FUNC int erase(const T val);
		DYN_FUNC void erase(iterator val_ptr);

	private:
		uint m_size = 0;
	};

}

#include "MultiSet.inl"

#endif // MultiSet_H
