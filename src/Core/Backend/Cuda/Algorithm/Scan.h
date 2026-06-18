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

		void exclusive(T* output, const T* input, uint length, bool bcao = true);
		void exclusive(T* data, uint length, bool bcao = true);

		void exclusive(DArray<T>& output, DArray<T>& input, bool bcao = true);
		void exclusive(DArray<T>& data, bool bcao = true);
		// bcao: memory-bank conflict avoidance optimization
	private:
		void scanLargeDeviceArray(T*d_out, const T*d_in, uint length, bool bcao, uint level);
		void scanSmallDeviceArray(T*d_out, const T*d_in, uint length, bool bcao);
		void scanLargeEvenDeviceArray(T*output, const T*input, uint length, bool bcao, uint level);

		bool isPowerOfTwo(uint x);
		uint nextPowerOfTwo(uint x);


	private:
		DArray<T> m_buffer;

		DArray<T> m_sums[SCAN_LEVEL];
		DArray<T> m_incr[SCAN_LEVEL];
	};

}


