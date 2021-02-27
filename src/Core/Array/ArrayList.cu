#pragma once
#include "ArrayList.h"

#include "Utility.h"

#include <thrust/sort.h>

namespace dyno
{
	template<typename ElementType>
	__global__ void DA_SetupListArray(
		GArray<List<ElementType>> lists,
		GArray<ElementType> elements,
		GArray<int> index)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= index.size()) return;

		int count = tId == index.size() - 1 ? elements.size() - index[index.size() - 1] : index[tId + 1] - index[tId];

		List<ElementType> list;
		list.reserve(elements.begin() + index[tId], count);

		lists[tId] = list;
	}

	template< class  ElementType, DeviceType deviceType /*= DeviceType::GPU*/>
	bool ArrayList<ElementType, deviceType>::allocate(GArray<int> counts)
	{
		if (index.size() != counts.size())
		{
			index.resize(counts.size());
			m_lists.resize(counts.size());
		}

		Function1Pt::copy(index, counts);

		int total_num = thrust::reduce(thrust::device, index.begin(), index.begin() + index.size());
		thrust::exclusive_scan(thrust::device, index.begin(), index.begin() + index.size(), index.begin());
		

		m_elements.resize(total_num);

		cuExecute(index.size(),
			DA_SetupListArray,
			m_lists,
			m_elements,
			index);

		return true;
	}
}