#include "SemiAnalyticalSharedFunc.h"

namespace dyno
{
	__global__ void K_SetupAttributesForSFI(
		DArray<Attribute> att)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= att.size()) return;

		att[tId].setFluid();
		att[tId].setDynamic();
	}

	void SetupAttributesForSFI(DArray<Attribute>& att)
	{
		cuExecute(att.size(),
			K_SetupAttributesForSFI,
			att);
	}
}