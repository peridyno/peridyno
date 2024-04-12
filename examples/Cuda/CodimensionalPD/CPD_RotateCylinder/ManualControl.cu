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
		: ComputeModule()
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

		if (abs(position[pId][2] - 1.5f) <= 5e-3 || abs(position[pId][2] + 1.5f) <= 5e-3)
		{
			Attribute& a = att[pId];
			a.setPassive();
		}
		
	}

	template< typename Coord>
	__global__ void UpdateVelocity(
		DArray<Coord> velocity,
		DArray<Coord> position,
		DArray<Attribute> atts)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= velocity.size()) return;

		Attribute att = atts[pId];

		
		if (att.isPassive()) {
			if (abs(position[pId][2] - 1.5f) <= 5e-3) {
				Coord p = Vec3f(0);
				p[0] = position[pId][0];
				p[1] = position[pId][1] - 1;
				velocity[pId] = p.cross(Coord(0, 0, 1));
			}
			else {
				if (abs(position[pId][2] + 1.5f) <= 5e-3)
				{
					Coord p = Vec3f(0);
					p[0] = position[pId][0];
					p[1] = position[pId][1] - 1;
					velocity[pId] = p.cross(Coord(0, 0, -1));
				}
			}
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
		cuExecute(this->inPosition()->getData().size(),
			UpdateVelocity,
			this->inVelocity()->getData(),
			this->inPosition()->getData(),
			this->inAttribute()->getData());
	}

#ifdef PRECISION_FLOAT
	template class ManualControl<DataType3f>;
#else
	template class ManualControl<DataType3d>;
#endif
}