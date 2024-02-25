#include "gtest/gtest.h"
#include "Array/ArrayList.h"
#include "STL/Heap.h"

using namespace std;
using namespace dyno;

__global__ void heap_test_GPU(DArray<float> test_gpu, uint numSize) {
	printf("================== gpu pop_heap test ===============\n");
	dyno::greater<float> cmp;
	dyno::make_heap<float*, dyno::greater<float>>(test_gpu.begin(), test_gpu.begin() + test_gpu.size(), cmp);
	for (int i = 0; i < numSize; i++)
	{
		printf("num %d: %f\n", i, test_gpu[0]);
		dyno::pop_heap<float*, dyno::greater<float>>(test_gpu.begin(), test_gpu.begin() + test_gpu.size() - i, cmp);
	}
	printf("=================== gpu sort_heap test ==============\n");
	dyno::sort_heap<float*, dyno::greater<float>>(test_gpu.begin(), test_gpu.begin(), cmp);
	for (int i = 0; i < numSize; i++)
	{
		printf("num %d: %f\n", i, test_gpu[i]);	
	}

}

TEST(Heap, sort)
{
	uint numSize = 20;
	CArray<float> test_cpu;
	DArray<float> test_gpu;
	test_cpu.resize(numSize);
	for (int i = 0; i < numSize; i++)
	{
		test_cpu[i] = (float)(i+sinf(i) * 10)* 0.1f;
		printf("original num %d: %f\n", i, test_cpu[i]);
	}

	test_gpu.assign(test_cpu);
	printf("=================== cpu pop_heap test ==============\n");
	dyno::greater<float> cmp;
	dyno::make_heap<float*, dyno::greater<float>>(test_cpu.begin(), test_cpu.begin() + test_cpu.size(), cmp);
	for (int i = 0; i < numSize; i++)
	{
		printf("num %d: %f\n", i, test_cpu[0]);
		dyno::pop_heap<float*, dyno::greater<float>>(test_cpu.begin(), test_cpu.begin() + test_cpu.size()-i, cmp);
		
	}

	printf("=================== cpu sort_heap test ==============\n");
	dyno::sort_heap<float*, dyno::greater<float>>(test_cpu.begin(), test_cpu.begin(), cmp);
	for (int i = 0; i < numSize; i++)
	{
		printf("num %d: %f\n", i, test_cpu[i]);
	
	}

	heap_test_GPU << < 1, 1 >> > (test_gpu, numSize);
	cudaDeviceSynchronize();
}


