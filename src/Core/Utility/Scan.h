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

		void exclusive(int* output, int* input, int length, bool bcao = true);
		void exclusive(int* data, int length, bool bcao = true);

		void exclusive(DeviceArray<int>& output, DeviceArray<int>& input, bool bcao = true);
		void exclusive(DeviceArray<int>& data, bool bcao = true);

	private:
		void scanLargeDeviceArray(int *d_out, int *d_in, int length, bool bcao, int level);
		void scanSmallDeviceArray(int *d_out, int *d_in, int length, bool bcao);
		void scanLargeEvenDeviceArray(int *output, int *input, int length, bool bcao, int level);

		bool isPowerOfTwo(int x);
		int nextPowerOfTwo(int x);


	private:
		DeviceArray<int> m_buffer;

		DeviceArray<int> m_sums[SCAN_LEVEL];
		DeviceArray<int> m_incr[SCAN_LEVEL];
	};

}


