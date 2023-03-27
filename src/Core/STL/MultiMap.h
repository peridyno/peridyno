#ifndef MultiMap_H
#define MultiMap_H

#include "Platform.h"
#include "Pair.h"

namespace dyno
{
	/**
	 * @brief An CPU/GPU implementation of the standard multimap suitable for small-size data
	 * 
	 * Be aware do not use this structure if the data size is large,
	 * because the computation complexity is O(n^2) for some specific situation.
	 * 
	 * All elements are organized in non-descending order.
	 * 
	 * @tparam T 
	 */
	template <typename MKey, typename T>
	class MultiMap
	{
	public:
		using iterator = Pair<MKey, T>*;

		DYN_FUNC MultiMap();
		
		DYN_FUNC void reserve(Pair<MKey, T>* buf, int maxSize) {
			m_startLoc = buf;
			m_maxSize = maxSize;
		}

		DYN_FUNC iterator find(MKey key);

		DYN_FUNC inline iterator begin() {
			return m_startLoc;
		};

		DYN_FUNC inline iterator end(){
			return m_startLoc + m_size;
		}

		DYN_FUNC void clear();

		DYN_FUNC uint size();
		DYN_FUNC uint count(MKey key);

		DYN_FUNC inline T& operator[] (MKey key);
		DYN_FUNC inline const T& operator[] (MKey key) const;

		DYN_FUNC iterator insert(Pair<MKey, T> pair);
		DYN_FUNC bool empty();

	private:
		uint m_size = 0;

		Pair<MKey, T>* m_startLoc = nullptr;
		uint m_maxSize = 0;
	};

}

#include "MultiMap.inl"

#endif // MultiSet_H
