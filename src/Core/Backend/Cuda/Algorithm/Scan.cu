#include "Scan.h"
#include "SharedMemory.h"

namespace dyno
{
#define SCAN_SHARED_MEMORY_BANKS 32
#define SCAN_LOG_MEM_BANKS 5
#define SCAN_CONFLICT_FREE_OFFSET(n) ((n) >> SCAN_LOG_MEM_BANKS)

	uint SCAN_THREADS_PER_BLOCK = 512;
	uint SCAN_ELEMENTS_PER_BLOCK = SCAN_THREADS_PER_BLOCK * 2;

	template<typename T>
	Scan<T>::Scan()
	{
	}

	template<typename T>
	Scan<T>::~Scan()
	{
		m_buffer.clear();

		for (int i = 0; i < SCAN_LEVEL; i++)
		{
			m_sums[i].clear();
			m_incr[i].clear();
		}
	}

	template<typename T>
	void Scan<T>::exclusive(DArray<T>& output, DArray<T>& input, bool bcao)
	{
		assert(input.size() == output.size());

		if (input.size() > SCAN_ELEMENTS_PER_BLOCK) {
			scanLargeDeviceArray(output.begin(), input.begin(), input.size(), bcao, 0);
		}
		else {
			scanSmallDeviceArray(output.begin(), input.begin(), input.size(), bcao);
		}
	}

	template<typename T>
	void Scan<T>::exclusive(DArray<T>& data, bool bcao /*= true*/)
	{
		if (m_buffer.size() != data.size())
		{
			m_buffer.resize(data.size());
		}
		
		m_buffer.assign(data);
		this->exclusive(data, m_buffer, bcao);
	}

	template<typename T>
	void Scan<T>::exclusive(T* data, size_t length, bool bcao /*= true*/)
	{
		if (m_buffer.size() != length)
		{
			m_buffer.resize(length);
		}

		cudaMemcpy(m_buffer.begin(), data, length*sizeof(T), cudaMemcpyDeviceToDevice);

		this->exclusive(data, m_buffer.begin(), length, bcao);
	}

	template<typename T>
	void Scan<T>::exclusive(T* output, T* input, size_t length, bool bcao /*= true*/)
	{
		if (length > SCAN_ELEMENTS_PER_BLOCK) {
			scanLargeDeviceArray(output, input, length, bcao, 0);
		}
		else {
			scanSmallDeviceArray(output, input, length, bcao);
		}
	}

	template<typename T>
	__global__ void k_prescan_arbitrary(T *output, T *input, size_t n, int powerOfTwo)
	{
		//extern __shared__ T temp[];// allocated on invocation
		SharedMemory<T> shared_mem;
		T* temp = shared_mem.getPointer();

		int threadID = threadIdx.x;

		int ai = threadID;
		int bi = threadID + (n / 2);
		int bankOffsetA = SCAN_CONFLICT_FREE_OFFSET(ai);
		int bankOffsetB = SCAN_CONFLICT_FREE_OFFSET(bi);


		if (threadID < n) {
			temp[ai + bankOffsetA] = input[ai];
			temp[bi + bankOffsetB] = input[bi];
		}
		else {
			temp[ai + bankOffsetA] = 0;
			temp[bi + bankOffsetB] = 0;
		}


		int offset = 1;
		for (int d = powerOfTwo >> 1; d > 0; d >>= 1) // build sum in place up the tree
		{
			__syncthreads();
			if (threadID < d)
			{
				int ai = offset * (2 * threadID + 1) - 1;
				int bi = offset * (2 * threadID + 2) - 1;
				ai += SCAN_CONFLICT_FREE_OFFSET(ai);
				bi += SCAN_CONFLICT_FREE_OFFSET(bi);

				temp[bi] += temp[ai];
			}
			offset *= 2;
		}

		if (threadID == 0) {
			temp[powerOfTwo - 1 + SCAN_CONFLICT_FREE_OFFSET(powerOfTwo - 1)] = 0; // clear the last element
		}

		for (int d = 1; d < powerOfTwo; d *= 2) // traverse down tree & build scan
		{
			offset >>= 1;
			__syncthreads();
			if (threadID < d)
			{
				int ai = offset * (2 * threadID + 1) - 1;
				int bi = offset * (2 * threadID + 2) - 1;
				ai += SCAN_CONFLICT_FREE_OFFSET(ai);
				bi += SCAN_CONFLICT_FREE_OFFSET(bi);

				int t = temp[ai];
				temp[ai] = temp[bi];
				temp[bi] += t;
			}
		}
		__syncthreads();

		if (threadID < n) {
			output[ai] = temp[ai + bankOffsetA];
			output[bi] = temp[bi + bankOffsetB];
		}
	}

	template<typename T>
	__global__ void k_prescan_arbitrary_unoptimized(T *output, T *input, size_t n, int powerOfTwo) {
		//extern __shared__ T temp[];// allocated on invocation
		SharedMemory<T> shared_mem;
		T* temp = shared_mem.getPointer();

		int threadID = threadIdx.x;

		if (threadID < n) {
			temp[2 * threadID] = input[2 * threadID]; // load input into shared memory
			temp[2 * threadID + 1] = input[2 * threadID + 1];
		}
		else {
			temp[2 * threadID] = 0;
			temp[2 * threadID + 1] = 0;
		}


		int offset = 1;
		for (int d = powerOfTwo >> 1; d > 0; d >>= 1) // build sum in place up the tree
		{
			__syncthreads();
			if (threadID < d)
			{
				int ai = offset * (2 * threadID + 1) - 1;
				int bi = offset * (2 * threadID + 2) - 1;
				temp[bi] += temp[ai];
			}
			offset *= 2;
		}

		if (threadID == 0) { temp[powerOfTwo - 1] = 0; } // clear the last element

		for (int d = 1; d < powerOfTwo; d *= 2) // traverse down tree & build scan
		{
			offset >>= 1;
			__syncthreads();
			if (threadID < d)
			{
				int ai = offset * (2 * threadID + 1) - 1;
				int bi = offset * (2 * threadID + 2) - 1;
				int t = temp[ai];
				temp[ai] = temp[bi];
				temp[bi] += t;
			}
		}
		__syncthreads();

		if (threadID < n) {
			output[2 * threadID] = temp[2 * threadID]; // write results to device memory
			output[2 * threadID + 1] = temp[2 * threadID + 1];
		}
	}

	template<typename T>
	__global__ void k_add(T *output, size_t length, T *n) {
		int blockID = blockIdx.x;
		int threadID = threadIdx.x;
		int blockOffset = blockID * length;

		output[blockOffset + threadID] += n[blockID];
	}

	template<typename T>
	__global__ void k_add(T *output, size_t length, T *n1, T *n2) {
		int blockID = blockIdx.x;
		int threadID = threadIdx.x;
		int blockOffset = blockID * length;

		output[blockOffset + threadID] += n1[blockID] + n2[blockID];
	}

	template<typename T>
	void Scan<T>::scanLargeDeviceArray(T *d_out, T *d_in, size_t length, bool bcao, size_t level)
	{
		size_t remainder = length % (SCAN_ELEMENTS_PER_BLOCK);
		if (remainder == 0) {
			scanLargeEvenDeviceArray(d_out, d_in, length, bcao, level);
		}
		else {
			// perform a large scan on a compatible multiple of elements
			size_t lengthMultiple = length - remainder;
			scanLargeEvenDeviceArray(d_out, d_in, lengthMultiple, bcao, level);

			// scan the remaining elements and add the (inclusive) last element of the large scan to this
			T *startOfOutputArray = &(d_out[lengthMultiple]);
			scanSmallDeviceArray(startOfOutputArray, &(d_in[lengthMultiple]), remainder, bcao);

			k_add << <1, (unsigned int)remainder >> > (startOfOutputArray, remainder, &(d_in[lengthMultiple - 1]), &(d_out[lengthMultiple - 1]));
			cuSynchronize();
		}
	}

	template<typename T>
	void Scan<T>::scanSmallDeviceArray(T *d_out, T *d_in, size_t length, bool bcao)
	{
		size_t powerOfTwo = nextPowerOfTwo(length);

		if (bcao) {
			k_prescan_arbitrary << <1, (unsigned int)(length + 1) / 2, (unsigned int)2 * powerOfTwo * sizeof(T) >> > (d_out, d_in, length, powerOfTwo);
			cuSynchronize();
		}
		else {
			k_prescan_arbitrary_unoptimized << <1, (unsigned int)(length + 1) / 2, (unsigned int)2 * powerOfTwo * sizeof(T) >> > (d_out, d_in, length, powerOfTwo);
			cuSynchronize();
		}
	}

	template<typename T>
	__global__ void k_prescan_large(T *output, T *input, int n, T *sums) {
		//extern __shared__ T temp[];
		SharedMemory<T> shared_mem;
		T* temp = shared_mem.getPointer();

		int blockID = blockIdx.x;
		int threadID = threadIdx.x;
		int blockOffset = blockID * n;

		int ai = threadID;
		int bi = threadID + (n / 2);
		int bankOffsetA = SCAN_CONFLICT_FREE_OFFSET(ai);
		int bankOffsetB = SCAN_CONFLICT_FREE_OFFSET(bi);
		temp[ai + bankOffsetA] = input[blockOffset + ai];
		temp[bi + bankOffsetB] = input[blockOffset + bi];

		int offset = 1;
		for (int d = n >> 1; d > 0; d >>= 1) // build sum in place up the tree
		{
			__syncthreads();
			if (threadID < d)
			{
				int ai = offset * (2 * threadID + 1) - 1;
				int bi = offset * (2 * threadID + 2) - 1;
				ai += SCAN_CONFLICT_FREE_OFFSET(ai);
				bi += SCAN_CONFLICT_FREE_OFFSET(bi);

				temp[bi] += temp[ai];
			}
			offset *= 2;
		}
		__syncthreads();


		if (threadID == 0) {
			sums[blockID] = temp[n - 1 + SCAN_CONFLICT_FREE_OFFSET(n - 1)];
			temp[n - 1 + SCAN_CONFLICT_FREE_OFFSET(n - 1)] = 0;
		}

		for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
		{
			offset >>= 1;
			__syncthreads();
			if (threadID < d)
			{
				int ai = offset * (2 * threadID + 1) - 1;
				int bi = offset * (2 * threadID + 2) - 1;
				ai += SCAN_CONFLICT_FREE_OFFSET(ai);
				bi += SCAN_CONFLICT_FREE_OFFSET(bi);

				int t = temp[ai];
				temp[ai] = temp[bi];
				temp[bi] += t;
			}
		}
		__syncthreads();

		output[blockOffset + ai] = temp[ai + bankOffsetA];
		output[blockOffset + bi] = temp[bi + bankOffsetB];
	}

	template<typename T>
	__global__ void k_prescan_large_unoptimized(T *output, T *input, int n, T *sums) {
		int blockID = blockIdx.x;
		int threadID = threadIdx.x;
		int blockOffset = blockID * n;

		//extern __shared__ T temp[];
		SharedMemory<T> shared_mem;
		T* temp = shared_mem.getPointer();

		temp[2 * threadID] = input[blockOffset + (2 * threadID)];
		temp[2 * threadID + 1] = input[blockOffset + (2 * threadID) + 1];

		int offset = 1;
		for (int d = n >> 1; d > 0; d >>= 1) // build sum in place up the tree
		{
			__syncthreads();
			if (threadID < d)
			{
				int ai = offset * (2 * threadID + 1) - 1;
				int bi = offset * (2 * threadID + 2) - 1;
				temp[bi] += temp[ai];
			}
			offset *= 2;
		}
		__syncthreads();


		if (threadID == 0) {
			sums[blockID] = temp[n - 1];
			temp[n - 1] = 0;
		}

		for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
		{
			offset >>= 1;
			__syncthreads();
			if (threadID < d)
			{
				int ai = offset * (2 * threadID + 1) - 1;
				int bi = offset * (2 * threadID + 2) - 1;
				int t = temp[ai];
				temp[ai] = temp[bi];
				temp[bi] += t;
			}
		}
		__syncthreads();

		output[blockOffset + (2 * threadID)] = temp[2 * threadID];
		output[blockOffset + (2 * threadID) + 1] = temp[2 * threadID + 1];
	}

	template<typename T>
	void Scan<T>::scanLargeEvenDeviceArray(T *output, T *input, size_t length, bool bcao, size_t level)
	{
		const int blocks = length / SCAN_ELEMENTS_PER_BLOCK;
		const int sharedMemArraySize = SCAN_ELEMENTS_PER_BLOCK * sizeof(T);

		//The following code is used to avoid malloc GPU memory for each call
		if (level < SCAN_LEVEL)
		{
			const int blocks = length / SCAN_ELEMENTS_PER_BLOCK;
			const int sharedMemArraySize = SCAN_ELEMENTS_PER_BLOCK * sizeof(T);

			if (m_sums[level].size() != blocks)
			{
				m_sums[level].resize(blocks);
				m_incr[level].resize(blocks);
			}

			if (bcao) {
				k_prescan_large << <blocks, SCAN_THREADS_PER_BLOCK, 2 * sharedMemArraySize >> > (output, input, SCAN_ELEMENTS_PER_BLOCK, m_sums[level].begin());
				cuSynchronize();
			}
			else {
				k_prescan_large_unoptimized << <blocks, SCAN_THREADS_PER_BLOCK, 2 * sharedMemArraySize >> > (output, input, SCAN_ELEMENTS_PER_BLOCK, m_sums[level].begin());
				cuSynchronize();
			}

			const int sumsArrThreadsNeeded = (blocks + 1) / 2;
			if (sumsArrThreadsNeeded > SCAN_THREADS_PER_BLOCK) {
				// perform a large scan on the sums arr
				scanLargeDeviceArray(m_incr[level].begin(), m_sums[level].begin(), blocks, bcao, level+1);
			}
			else {
				// only need one block to scan sums arr so can use small scan
				scanSmallDeviceArray(m_incr[level].begin(), m_sums[level].begin(), blocks, bcao);
			}

			k_add << <blocks, SCAN_ELEMENTS_PER_BLOCK >> > (output, SCAN_ELEMENTS_PER_BLOCK, m_incr[level].begin());
			cuSynchronize();
		}
		else
		{
			T *d_sums, *d_incr;
			cudaMalloc((void **)&d_sums, blocks * sizeof(T));
			cudaMalloc((void **)&d_incr, blocks * sizeof(T));

			if (bcao) {
				k_prescan_large << <blocks, SCAN_THREADS_PER_BLOCK, 2 * sharedMemArraySize >> > (output, input, SCAN_ELEMENTS_PER_BLOCK, d_sums);
				cuSynchronize();
			}
			else {
				k_prescan_large_unoptimized << <blocks, SCAN_THREADS_PER_BLOCK, 2 * sharedMemArraySize >> > (output, input, SCAN_ELEMENTS_PER_BLOCK, d_sums);
				cuSynchronize();
			}

			const int sumsArrThreadsNeeded = (blocks + 1) / 2;
			if (sumsArrThreadsNeeded > SCAN_THREADS_PER_BLOCK) {
				// perform a large scan on the sums arr
				scanLargeDeviceArray(d_incr, d_sums, blocks, bcao, level + 1);
			}
			else {
				// only need one block to scan sums arr so can use small scan
				scanSmallDeviceArray(d_incr, d_sums, blocks, bcao);
			}

			k_add << <blocks, SCAN_ELEMENTS_PER_BLOCK >> > (output, SCAN_ELEMENTS_PER_BLOCK, d_incr);
			cuSynchronize();

			cudaFree(d_sums);
			cudaFree(d_incr);
		}
	}

	template<typename T>
	bool Scan<T>::isPowerOfTwo(size_t x)
	{
		return x && !(x & (x - 1));
	}

	template<typename T>
	size_t Scan<T>::nextPowerOfTwo(size_t x)
	{
		int power = 1;
		while (power < x) {
			power *= 2;
		}
		return power;
	}

	template class Scan<int>;
	template class Scan<uint>;
}

