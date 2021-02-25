#include <cuda_runtime.h>
#include "Framework/Node.h"
#include "VelocityControl.h"

namespace dyno
{
	template<typename TDataType>
	VelocityControl<TDataType>::VelocityControl()
		: CustomModule()
	{

	}

	template<typename TDataType>
	VelocityControl<TDataType>::~VelocityControl()
	{
	}

	template<typename Coord>
	__global__ void Constrain(
		DeviceArray<Coord> poss,
		DeviceArray<Coord> vels)
	{
		int vId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (vId >= poss.size()) return;

		Coord vel = vels[vId];

		vels[vId] = vel*0.999;
	}

	int i = 0;
	template<typename TDataType>
	void VelocityControl<TDataType>::applyCustomBehavior()
	{
		auto& poss = this->inPosition()->getValue();
		auto& vels = this->inVelocity()->getValue();
		
		cuExecute(poss.size(),
			Constrain,
			poss,
			vels);
	}
}