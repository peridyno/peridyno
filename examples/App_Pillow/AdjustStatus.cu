#include <cuda_runtime.h>
#include "Framework/Node.h"
#include "AdjustStatus.h"

#include "Utility/CudaRand.h"

namespace dyno
{
	template<typename TDataType>
	AdjustStatus<TDataType>::AdjustStatus()
		: CustomModule()
	{

	}

	template<typename TDataType>
	AdjustStatus<TDataType>::~AdjustStatus()
	{
	}

	template<typename Coord>
	__global__ void SetAttribute(
		DeviceArray<Coord> poss,
		DeviceArray<Attribute> atts)
	{
		int vId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (vId >= atts.size()) return;

		Coord pos = poss[vId];

		RandNumber rNum(vId);

		poss[vId] = pos + 0.09*rNum.Generate();

// 		Real space = 0.025;
// 		if (pos[1] > 2 - space || pos[1] < -2 + space || pos[2] > 2 - space || pos[2] < -2 + space)
// 		{
// 			atts[vId].SetFixed();
// 		}
	}


	template<typename Coord>
	__global__ void Constrain(
		DeviceArray<Coord> poss,
		DeviceArray<Coord> vels)
	{
		int vId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (vId >= poss.size()) return;

		Coord vel = vels[vId];

		vels[vId] = vel * 0.8;
	}


	int i = 0;
	template<typename TDataType>
	void AdjustStatus<TDataType>::applyCustomBehavior()
	{
		auto& atts = this->inVertexAttribute()->getValue();
		auto& poss = this->inPosition()->getValue();
		auto& vels = this->inVelocity()->getValue();

		if(i == 2)
			cuExecute(atts.size(),
				SetAttribute,
				poss,
				atts);

		cuExecute(poss.size(),
			Constrain,
			poss,
			vels);

		i++;
	}
}