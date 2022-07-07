#include "LinearDamping.h"

namespace dyno
{
//	IMPLEMENT_TCLASS(LinearDamping, TDataType)

	template<typename TDataType>
	LinearDamping<TDataType>::LinearDamping()
		: ConstraintModule()
	{
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

	template<typename TDataType>
	void LinearDamping<TDataType>::constrain()
	{
		Real coef = this->varDampingCoefficient()->getData();
		auto& vels = this->inVelocity()->getData();

		int num = vels.size();
		cuExecute(num,
			LP_Damping,
			vels,
			coef);
	}

	DEFINE_CLASS(LinearDamping);
}