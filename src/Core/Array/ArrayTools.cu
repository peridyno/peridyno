#include "ArrayTools.h"
#include "STL/List.h"

namespace dyno
{
	template<int N>
	__global__ void AA_Allocate(
		void* lists,
		void* elements,
		size_t ele_size,
		GArray<int> index)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= index.size()) return;

		List<PlaceHolder<N>>* listPtr = (List<PlaceHolder<N>>*)lists;
		PlaceHolder<N>* elementsPtr = (PlaceHolder<N>*)elements;

		int count = tId == index.size() - 1 ? ele_size - index[index.size() - 1] : index[tId + 1] - index[tId];

		List<PlaceHolder<N>> list;
		list.reserve(elementsPtr + index[tId], count);

		listPtr[tId] = list;
	}

	template<int N>
	void parallel_allocate_for_list(void* lists, void* elements, size_t ele_size, GArray<int> index)
	{
		uint pDims = cudaGridSize(index.size(), BLOCK_SIZE);
		AA_Allocate<N> << <pDims, BLOCK_SIZE >> > (	
			lists,
			elements,
			ele_size,
			index);
		cuSynchronize();
	}

	template void parallel_allocate_for_list<1>(void* lists, void* elements, size_t ele_size, GArray<int> index);
	template void parallel_allocate_for_list<2>(void* lists, void* elements, size_t ele_size, GArray<int> index);
	template void parallel_allocate_for_list<3>(void* lists, void* elements, size_t ele_size, GArray<int> index);
	template void parallel_allocate_for_list<4>(void* lists, void* elements, size_t ele_size, GArray<int> index);
	template void parallel_allocate_for_list<5>(void* lists, void* elements, size_t ele_size, GArray<int> index);
	template void parallel_allocate_for_list<6>(void* lists, void* elements, size_t ele_size, GArray<int> index);
	template void parallel_allocate_for_list<7>(void* lists, void* elements, size_t ele_size, GArray<int> index);
	template void parallel_allocate_for_list<8>(void* lists, void* elements, size_t ele_size, GArray<int> index);

	template<int N>
	__global__ void AA_Assign(
		void* lists,
		void* elements,
		size_t ele_size,
		GArray<int> index)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= index.size()) return;

		List<PlaceHolder<N>>* listStartPtr = (List<PlaceHolder<N>>*)lists;
		PlaceHolder<N>* elementsPtr = (PlaceHolder<N>*)elements;

		int count = tId == index.size() - 1 ? ele_size - index[index.size() - 1] : index[tId + 1] - index[tId];

		List<PlaceHolder<N>> list = *(listStartPtr + tId);
		list.reserve(elementsPtr + index[tId], count);

		listStartPtr[tId] = list;
	}

	template<int N>
	void parallel_init_for_list(void* lists, void* elements, size_t ele_size, GArray<int> index)
	{
		uint pDims = cudaGridSize(index.size(), BLOCK_SIZE);
		AA_Assign<N> << <pDims, BLOCK_SIZE >> > (
			lists,
			elements,
			ele_size,
			index);
		cuSynchronize();
	}

	template void parallel_init_for_list<1>(void* lists, void* elements, size_t ele_size, GArray<int> index);
	template void parallel_init_for_list<2>(void* lists, void* elements, size_t ele_size, GArray<int> index);
	template void parallel_init_for_list<3>(void* lists, void* elements, size_t ele_size, GArray<int> index);
	template void parallel_init_for_list<4>(void* lists, void* elements, size_t ele_size, GArray<int> index);
	template void parallel_init_for_list<5>(void* lists, void* elements, size_t ele_size, GArray<int> index);
	template void parallel_init_for_list<6>(void* lists, void* elements, size_t ele_size, GArray<int> index);
	template void parallel_init_for_list<7>(void* lists, void* elements, size_t ele_size, GArray<int> index);
	template void parallel_init_for_list<8>(void* lists, void* elements, size_t ele_size, GArray<int> index);
}