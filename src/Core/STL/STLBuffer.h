#ifndef STLBUFFER_H
#define STLBUFFER_H

#include "Platform.h"

namespace dyno
{
	/**
	 * @brief Be aware do not use this structure on GPU if the data size is large.
	 * 
	 * @tparam T 
	 */
	template <typename T>
	class STLBuffer
	{
	public:
		using iterator = T * ;

		DYN_FUNC STLBuffer() {};
		
		DYN_FUNC void reserve(T* beg, uint buffer_size) {
			m_startLoc = beg;
			m_maxSize = buffer_size;
		}

		DYN_FUNC uint max_size() { return m_maxSize; }

// 		DYN_FUNC inline T& operator[] (unsigned int) { return m_startLoc[i]; }
// 		DYN_FUNC inline const T& operator[] (unsigned int) const { return m_startLoc[i]; }

	protected:
		DYN_FUNC inline T* bufferEnd() {
			return m_startLoc + m_maxSize;
		}

		uint m_maxSize = 0;
		
		T* m_startLoc = nullptr;
	};
}

#endif // STLBUFFER_H
