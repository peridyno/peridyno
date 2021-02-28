#include "Allocator.h"
#include "Utility/cuda_utilities.h"

namespace dyno
{
	template<int N>
	__global__ void AA_Allocate(
		void* lists,
		void* elements,
		int ele_size,
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
	void parallel_allocate_for_list(void* lists, void* elements, int ele_size, GArray<int> index)
	{
		uint pDims = cudaGridSize(index.size(), BLOCK_SIZE);
		AA_Allocate<N> << <pDims, BLOCK_SIZE >> > (	
			lists,
			elements,
			ele_size,
			index);
		cuSynchronize();
	}


}