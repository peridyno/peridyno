#include "Function1Pt.h"
#include "Utility.h"
namespace dyno
{
	namespace Function1Pt
	{
		template<typename T1, typename T2>
		__global__ void KerLength(T1* lhs, T2* rhs, int num)
		{
			int pId = threadIdx.x + (blockIdx.x * blockDim.x);
			if (pId >= num) return;

			lhs[pId] = length(rhs[pId]);
		}

		template<typename T1, typename T2>
		void Length(GArray<T1>& lhs, GArray<T2>& rhs)
		{
			assert(lhs.size() == rhs.size());
			unsigned pDim = cudaGridSize(rhs.size(), BLOCK_SIZE);
			KerLength << <pDim, BLOCK_SIZE >> > (lhs.begin(), rhs.begin(), lhs.size());
		}

		template void Length(GArray<float>&, GArray<float3>&);
	}
}