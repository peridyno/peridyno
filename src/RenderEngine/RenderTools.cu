#include "RenderTools.h"

namespace dyno {

	__global__ void RT_SetupColor(DArray<Vec3f> colorArray, Vec3f color)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= colorArray.size()) return;

		colorArray[pId] = color;
	}

	void RenderTools::setupColor(DArray<Vec3f>& colorArray, const Vec3f& color)
	{
		cuExecute(colorArray.size(),
			RT_SetupColor,
			colorArray,
			color);
	}
}
