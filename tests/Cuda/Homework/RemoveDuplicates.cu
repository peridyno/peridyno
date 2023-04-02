#include "gtest/gtest.h"
#include "Array/Array.h"

#include "Algorithm/Reduction.h"
#include "Algorithm/Scan.h"

#include "Timer.h"

#include <thrust/sort.h>
#include <thrust/functional.h>

#include <vector>

using namespace dyno;

__global__ void K_CountNumber(
	DArray<uint> counter,
	DArray<uint> dArray)
{
	int tId = threadIdx.x + (blockIdx.x * blockDim.x);
	if (tId >= dArray.size()) return;

	//TODO: replace your algorithm
	counter[tId] = 0;
}

__global__ void K_RemoveDuplicates(
	DArray<uint> output,
	DArray<uint> input,
	DArray<uint> radix)
{
	int tId = threadIdx.x + (blockIdx.x * blockDim.x);
	if (tId >= input.size()) return;

	//TODO: replace your algorithm
}

TEST(RemoveDuplicates, GPU)
{
	//Initialize an array on the host
	CArray<uint> hArr;
	
	for (uint i = 0; i < 1000000; i++)
		hArr.pushBack(i % 20);

	DArray<uint> dArr;

	//Copy the array to the device
	dArr.assign(hArr);

	//Sort the array
	thrust::sort(thrust::device, dArr.begin(), dArr.begin() + dArr.size(), thrust::less<uint>());

	DArray<uint> counter(dArr.size());

	uint blockSize = 128;
	uint pDims = cudaGridSize(dArr.size(), blockSize);

	GTimer timer;
	timer.start();

	K_CountNumber << <pDims, blockSize>> > (counter, dArr);

	Reduction<uint> reduce;
	Scan<uint> scan;

	uint total = reduce.accumulate(counter.begin(), counter.size());
	scan.exclusive(counter);

	DArray<uint> output(total);

	K_RemoveDuplicates << <pDims, blockSize >> > (
		output,
		dArr,
		counter);

	timer.stop();

	std::cout << "Time Cost: " << timer.getElapsedTime() << " milliseconds " << std::endl;

	//Clear
	counter.clear();
	dArr.clear();
	hArr.clear();

	EXPECT_EQ(total == 20, true);
}