#include <cuda_runtime.h>
#include "Framework/Node.h"
#include "AdjustBoard.h"

namespace dyno
{
	template<typename TDataType>
	AdjustBoard<TDataType>::AdjustBoard()
		: CustomModule()
	{

	}

	template<typename TDataType>
	AdjustBoard<TDataType>::~AdjustBoard()
	{
	}

	template<typename Coord>
	__global__ void SetAttribute(
		GArray<Coord> poss,
		GArray<Attribute> atts)
	{
		int vId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (vId >= atts.size()) return;

		Coord pos = poss[vId];

		Real space = 0.025;
		if (pos[1] > 2 - space || pos[1] < -2 + space || pos[2] > 2 - space || pos[2] < -2 + space)
		{
			atts[vId].SetFixed();
		}
	}


	template<typename Coord>
	__global__ void Constrain(
		GArray<Coord> poss,
		GArray<Coord> vels)
	{
		int vId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (vId >= poss.size()) return;

		Coord vel = vels[vId];

		vels[vId] = vel * 0.9995;
	}


	int i = 0;
	template<typename TDataType>
	void AdjustBoard<TDataType>::applyCustomBehavior()
	{
		auto& atts = this->inVertexAttribute()->getValue();
		auto& poss = this->inPosition()->getValue();
		auto& vels = this->inVelocity()->getValue();

		if(i == 0)
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