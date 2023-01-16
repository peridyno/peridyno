#pragma once
#include "Array/Array.h"

namespace dyno
{
#define SCAN_LEVEL 2

	template<typename T>
	class Scan
	{
	public:
		Scan();
		~Scan();

		void exclusive(T* output, T* input, size_t length, bool bcao = true);
		void exclusive(T* data, size_t length, bool bcao = true);

		void exclusive(DArray<T>& output, DArray<T>& input, bool bcao = true);
		void exclusive(DArray<T>& data, bool bcao = true);

	private:
		void scanLargeDeviceArray(T*d_out, T*d_in, size_t length, bool bcao, size_t level);
		void scanSmallDeviceArray(T*d_out, T*d_in, size_t length, bool bcao);
		void scanLargeEvenDeviceArray(T*output, T*input, size_t length, bool bcao, size_t level);

		bool isPowerOfTwo(size_t x);
		size_t nextPowerOfTwo(size_t x);


	private:
		DArray<T> m_buffer;

		DArray<T> m_sums[SCAN_LEVEL];
		DArray<T> m_incr[SCAN_LEVEL];
	};

}


