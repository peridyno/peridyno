#include "LinearDamping.h"

namespace dyno
{
//	IMPLEMENT_TCLASS(LinearDamping, TDataType)

	template<typename TDataType>
	LinearDamping<TDataType>::LinearDamping()
		: ConstraintModule()
	{
		this->inAttribute()->tagOptional(true);
	}

	template<typename TDataType>
	LinearDamping<TDataType>::~LinearDamping()
	{
	}

	template <typename Real, typename Coord>
	__global__ void LP_Damping(
		DArray<Coord> vel,
		Real coefficient)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= vel.size()) return;
		vel[pId] *= coefficient;
	}

	template <typename Real, typename Coord, typename Attribute>
	__global__ void LP_Damping(
		DArray<Coord> vel,
		DArray<Attribute> attr,
		Real coefficient)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= vel.size()) return;
		if (attr[pId].isDynamic())
			vel[pId] *= coefficient;
	}

	template<typename TDataType>
	void LinearDamping<TDataType>::constrain()
	{
		Real coef = this->varDampingCoefficient()->getData();
		auto& vels = this->inVelocity()->getData();

		int num = vels.size();
		if (this->inAttribute()->isEmpty())
		{
			cuExecute(num,
				LP_Damping,
				vels,
				coef);
		}
		else
		{
  			cuExecute(num,
				LP_Damping,
				vels,
				this->inAttribute()->getData(),
				coef);
		}
	}

	DEFINE_CLASS(LinearDamping);
}