#include <cuda_runtime.h>
#include "Framework/Node.h"
#include "FixBoundary.h"

namespace dyno
{
	template<typename TDataType>
	FixBoundary<TDataType>::FixBoundary()
		: CustomModule()
	{

	}

	template<typename TDataType>
	FixBoundary<TDataType>::~FixBoundary()
	{
	}

	template<typename Coord>
	__device__ bool Inside(Coord pos)
	{
		Real space = -0.005;

		if (pos[0] > 1.25 - space || pos[0] < -1.25 + space || pos[2] > 2.5 - space || pos[2] < -2.5 + space)
		{
			return false;
		}

		return true;
	}

	template<typename Coord>
	__global__ void SetAttribute(
		DeviceArray<Coord> poss,
		DeviceArray<Attribute> atts)
	{
		int vId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (vId >= atts.size()) return;

		Coord pos = poss[vId];

		Real space = 0.005;
		if (pos[0] > 1.25 - space || pos[0] < -1.25 + space || pos[2] > 2.5 - space || pos[2] < -2.5 + space)
		{
			atts[vId].SetFixed();
		}
	}

	template<typename Coord>
	__global__ void Constrain(
		DeviceArray<Coord> poss,
		DeviceArray<Coord> vels)
	{
		int vId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (vId >= poss.size()) return;

		Coord pos = poss[vId];
		Coord vel = vels[vId];

		if (Inside(pos))
		{
			if (pos[1] < 0)
			{
				pos[1] = 0;
				vel = Coord(0, 0, 0);
			}
		}
		else
		{
			if (pos[1] < 0.02)
			{
				pos[1] = 0.02;
				vel = Coord(0, 0, 0);
			}
		}

		poss[vId] = pos;
		vels[vId] = vel*0.998;
	}

	int i = 0;
	template<typename TDataType>
	void FixBoundary<TDataType>::applyCustomBehavior()
	{
		auto& atts = this->inVertexAttribute()->getValue();
		auto& poss = this->inPosition()->getValue();
		auto& vels = this->inVelocity()->getValue();

		if (i == 0)
		{
			cuExecute(atts.size(),
				SetAttribute,
				poss,
				atts);
		}
		
		cuExecute(atts.size(),
			Constrain,
			poss,
			vels);

		i++;
	}
}