#include <cuda_runtime.h>
#include <Node.h>
#include "ManualControl.h"
#define maximum(a,b) a>b?a:b
#define minimum(a,b) a<b?a:b
namespace dyno
{
	IMPLEMENT_TCLASS(ManualControl, TDataType)

	template<typename TDataType>
	ManualControl<TDataType>::ManualControl()
		: CustomModule()
	{

	}

	template<typename TDataType>
	ManualControl<TDataType>::~ManualControl()
	{
	}

	template<typename Coord>
	__global__ void InitAttribute(
		DArray<Coord> position,
		DArray<Coord> velocity,
		DArray<Attribute>att)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;

		if (position[pId][0] < -0.5)
		{

			velocity[pId] = Coord(15, 0, 0);

		}
		else if (abs(position[pId][1] - 0.0f) <= 5e-3 ||
			abs(position[pId][1] - 2.5f) <= 5e-3 ||
			abs(position[pId][2] + 1.25f) <= 5e-3 ||
			abs(position[pId][2] - 1.25f) <= 5e-3) {
			Attribute& a = att[pId];
			a.setFixed();
		}
	}

	template<typename TDataType>
	void ManualControl<TDataType>::begin()
	{
		if (this->inFrameNumber()->getData() < 1) {
			cuExecute(this->inPosition()->getData().size(),
				InitAttribute,
				this->inPosition()->getData(),
				this->inVelocity()->getData(),
				this->inAttribute()->getData());
		}
	
	}

	template<typename TDataType>
	void ManualControl<TDataType>::applyCustomBehavior()
	{
		
	}

#ifdef PRECISION_FLOAT
	template class ManualControl<DataType3f>;
#else
	template class ManualControl<DataType3d>;
#endif
}