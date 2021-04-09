#include "ArrayTools.h"
#include "STL/List.h"

namespace dyno
{
	template<int N>
	__global__ void AA_Allocate(
		void* lists,
		void* elements,
		size_t ele_size,
		DArray<int> index)
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
	void parallel_allocate_for_list(void* lists, void* elements, size_t ele_size, DArray<int>& index)
	{
		uint pDims = cudaGridSize(index.size(), BLOCK_SIZE);
		AA_Allocate<N> << <pDims, BLOCK_SIZE >> > (	
			lists,
			elements,
			ele_size,
			index);
		cuSynchronize();
	}

	template void parallel_allocate_for_list<1>(void* lists, void* elements, size_t ele_size, DArray<int>& index);
	template void parallel_allocate_for_list<2>(void* lists, void* elements, size_t ele_size, DArray<int>& index);
	template void parallel_allocate_for_list<3>(void* lists, void* elements, size_t ele_size, DArray<int>& index);
	template void parallel_allocate_for_list<4>(void* lists, void* elements, size_t ele_size, DArray<int>& index);
	template void parallel_allocate_for_list<5>(void* lists, void* elements, size_t ele_size, DArray<int>& index);
	template void parallel_allocate_for_list<6>(void* lists, void* elements, size_t ele_size, DArray<int>& index);
	template void parallel_allocate_for_list<7>(void* lists, void* elements, size_t ele_size, DArray<int>& index);
	template void parallel_allocate_for_list<8>(void* lists, void* elements, size_t ele_size, DArray<int>& index);

	template void parallel_allocate_for_list<9>(void* lists, void* elements, size_t ele_size, DArray<int>& index);
	template void parallel_allocate_for_list<10>(void* lists, void* elements, size_t ele_size, DArray<int>& index);
	template void parallel_allocate_for_list<11>(void* lists, void* elements, size_t ele_size, DArray<int>& index);
	template void parallel_allocate_for_list<12>(void* lists, void* elements, size_t ele_size, DArray<int>& index);
	template void parallel_allocate_for_list<13>(void* lists, void* elements, size_t ele_size, DArray<int>& index);
	template void parallel_allocate_for_list<14>(void* lists, void* elements, size_t ele_size, DArray<int>& index);
	template void parallel_allocate_for_list<15>(void* lists, void* elements, size_t ele_size, DArray<int>& index);
	template void parallel_allocate_for_list<16>(void* lists, void* elements, size_t ele_size, DArray<int>& index);

	template void parallel_allocate_for_list<17>(void* lists, void* elements, size_t ele_size, DArray<int>& index);
	template void parallel_allocate_for_list<18>(void* lists, void* elements, size_t ele_size, DArray<int>& index);
	template void parallel_allocate_for_list<19>(void* lists, void* elements, size_t ele_size, DArray<int>& index);
	template void parallel_allocate_for_list<20>(void* lists, void* elements, size_t ele_size, DArray<int>& index);
	template void parallel_allocate_for_list<21>(void* lists, void* elements, size_t ele_size, DArray<int>& index);
	template void parallel_allocate_for_list<22>(void* lists, void* elements, size_t ele_size, DArray<int>& index);
	template void parallel_allocate_for_list<23>(void* lists, void* elements, size_t ele_size, DArray<int>& index);
	template void parallel_allocate_for_list<24>(void* lists, void* elements, size_t ele_size, DArray<int>& index);

	template void parallel_allocate_for_list<25>(void* lists, void* elements, size_t ele_size, DArray<int>& index);
	template void parallel_allocate_for_list<26>(void* lists, void* elements, size_t ele_size, DArray<int>& index);
	template void parallel_allocate_for_list<27>(void* lists, void* elements, size_t ele_size, DArray<int>& index);
	template void parallel_allocate_for_list<28>(void* lists, void* elements, size_t ele_size, DArray<int>& index);
	template void parallel_allocate_for_list<29>(void* lists, void* elements, size_t ele_size, DArray<int>& index);
	template void parallel_allocate_for_list<30>(void* lists, void* elements, size_t ele_size, DArray<int>& index);
	template void parallel_allocate_for_list<31>(void* lists, void* elements, size_t ele_size, DArray<int>& index);
	template void parallel_allocate_for_list<32>(void* lists, void* elements, size_t ele_size, DArray<int>& index);

	template<int N>
	__global__ void AA_Assign(
		void* lists,
		void* elements,
		size_t ele_size,
		DArray<int> index)
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
	void parallel_init_for_list(void* lists, void* elements, size_t ele_size, DArray<int>& index)
	{
		uint pDims = cudaGridSize(index.size(), BLOCK_SIZE);
		AA_Assign<N> << <pDims, BLOCK_SIZE >> > (
			lists,
			elements,
			ele_size,
			index);
		cuSynchronize();
	}

	template void parallel_init_for_list<1>(void* lists, void* elements, size_t ele_size, DArray<int>& index);
	template void parallel_init_for_list<2>(void* lists, void* elements, size_t ele_size, DArray<int>& index);
	template void parallel_init_for_list<3>(void* lists, void* elements, size_t ele_size, DArray<int>& index);
	template void parallel_init_for_list<4>(void* lists, void* elements, size_t ele_size, DArray<int>& index);
	template void parallel_init_for_list<5>(void* lists, void* elements, size_t ele_size, DArray<int>& index);
	template void parallel_init_for_list<6>(void* lists, void* elements, size_t ele_size, DArray<int>& index);
	template void parallel_init_for_list<7>(void* lists, void* elements, size_t ele_size, DArray<int>& index);
	template void parallel_init_for_list<8>(void* lists, void* elements, size_t ele_size, DArray<int>& index);

	template void parallel_init_for_list<9>(void* lists, void* elements, size_t ele_size, DArray<int>& index);
	template void parallel_init_for_list<10>(void* lists, void* elements, size_t ele_size, DArray<int>& index);
	template void parallel_init_for_list<11>(void* lists, void* elements, size_t ele_size, DArray<int>& index);
	template void parallel_init_for_list<12>(void* lists, void* elements, size_t ele_size, DArray<int>& index);
	template void parallel_init_for_list<13>(void* lists, void* elements, size_t ele_size, DArray<int>& index);
	template void parallel_init_for_list<14>(void* lists, void* elements, size_t ele_size, DArray<int>& index);
	template void parallel_init_for_list<15>(void* lists, void* elements, size_t ele_size, DArray<int>& index);
	template void parallel_init_for_list<16>(void* lists, void* elements, size_t ele_size, DArray<int>& index);

	template void parallel_init_for_list<17>(void* lists, void* elements, size_t ele_size, DArray<int>& index);
	template void parallel_init_for_list<18>(void* lists, void* elements, size_t ele_size, DArray<int>& index);
	template void parallel_init_for_list<19>(void* lists, void* elements, size_t ele_size, DArray<int>& index);
	template void parallel_init_for_list<20>(void* lists, void* elements, size_t ele_size, DArray<int>& index);
	template void parallel_init_for_list<21>(void* lists, void* elements, size_t ele_size, DArray<int>& index);
	template void parallel_init_for_list<22>(void* lists, void* elements, size_t ele_size, DArray<int>& index);
	template void parallel_init_for_list<23>(void* lists, void* elements, size_t ele_size, DArray<int>& index);
	template void parallel_init_for_list<24>(void* lists, void* elements, size_t ele_size, DArray<int>& index);

	template void parallel_init_for_list<25>(void* lists, void* elements, size_t ele_size, DArray<int>& index);
	template void parallel_init_for_list<26>(void* lists, void* elements, size_t ele_size, DArray<int>& index);
	template void parallel_init_for_list<27>(void* lists, void* elements, size_t ele_size, DArray<int>& index);
	template void parallel_init_for_list<28>(void* lists, void* elements, size_t ele_size, DArray<int>& index);
	template void parallel_init_for_list<29>(void* lists, void* elements, size_t ele_size, DArray<int>& index);
	template void parallel_init_for_list<30>(void* lists, void* elements, size_t ele_size, DArray<int>& index);
	template void parallel_init_for_list<31>(void* lists, void* elements, size_t ele_size, DArray<int>& index);
	template void parallel_init_for_list<32>(void* lists, void* elements, size_t ele_size, DArray<int>& index);
}