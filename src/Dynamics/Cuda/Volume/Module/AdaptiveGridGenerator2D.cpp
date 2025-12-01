#include "AdaptiveGridGenerator2D.h"
#include "Algorithm/CudaRand.h"

namespace dyno
{
	template<typename TDataType>
	AdaptiveGridGenerator2D<TDataType>::AdaptiveGridGenerator2D()
		: ComputeModule()
	{
	}

	template<typename TDataType>
	AdaptiveGridGenerator2D<TDataType>::~AdaptiveGridGenerator2D()
	{
	}

	DEFINE_CLASS(AdaptiveGridGenerator2D);
}