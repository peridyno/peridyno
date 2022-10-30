#pragma once
#include "Array/Array.h"

namespace dyno
{
#define SCAN_LEVEL 2

	class Scan
	{
	public:
		Scan();
		~Scan();

		void exclusive(int* output, int* input, size_t length, bool bcao = true);
		void exclusive(int* data, size_t length, bool bcao = true);

		void exclusive(DArray<int>& output, DArray<int>& input, bool bcao = true);
		void exclusive(DArray<int>& data, bool bcao = true);

	private:
		void scanLargeDeviceArray(int *d_out, int *d_in, size_t length, bool bcao, size_t level);
		void scanSmallDeviceArray(int *d_out, int *d_in, size_t length, bool bcao);
		void scanLargeEvenDeviceArray(int *output, int *input, size_t length, bool bcao, size_t level);

		bool isPowerOfTwo(size_t x);
		size_t nextPowerOfTwo(size_t x);


	private:
		DArray<int> m_buffer;

		DArray<int> m_sums[SCAN_LEVEL];
		DArray<int> m_incr[SCAN_LEVEL];
	};

}


