#include "FractureModule.h"
#include "ParticleSystem/SummationDensity.h"
#include "ParticleSystem/Kernel.h"

namespace dyno
{
	template<typename TDataType>
	FractureModule<TDataType>::FractureModule()
		: ElastoplasticityModule<TDataType>()
	{
		this->setCohesion(0.001);
	}


	template <typename Real, typename Coord, typename NPair>
	__global__ void PM_ComputeInvariants(
		DArray<Real> bulk_stiffiness,
		DArray<Coord> position,
		DArrayList<NPair> restShape,
		Real horizon,
		Real A,
		Real B,
		Real mu,
		Real lambda)
	{
		int i = threadIdx.x + (blockIdx.x * blockDim.x);
		if (i >= position.size()) return;

		CorrectedKernel<Real> kernSmooth;

		Real s_A = A;

		List<NPair>& rest_shape_i = restShape[i];
		Coord rest_pos_i = rest_shape_i[0].pos;
		Coord cur_pos_i = position[i];

		Real I1_i = 0.0f;
		Real J2_i = 0.0f;
		//compute the first and second invariants of the deformation state, i.e., I1 and J2
		int size_i = rest_shape_i.size();
		Real total_weight = Real(0);
		for (int ne = 1; ne < size_i; ne++)
		{
			NPair np_j = rest_shape_i[ne];
			Coord rest_pos_j = np_j.pos;
			int j = np_j.index;
			Real r = (rest_pos_i - rest_pos_j).norm();

			if (r > 0.01*horizon)
			{
				Real weight = kernSmooth.Weight(r, horizon);
				Coord p = (position[j] - cur_pos_i);
				Real ratio_ij = p.norm() / r;

				I1_i += weight*ratio_ij;

				total_weight += weight;
			}
		}

		if (total_weight > EPSILON)
		{
			I1_i /= total_weight;
		}
		else
		{
			I1_i = 1.0f;
		}

		for (int ne = 1; ne < size_i; ne++)
		{
			NPair np_j = rest_shape_i[ne];
			int j = np_j.index;
			Coord rest_pos_j = np_j.pos;
			Real r = (rest_pos_i - rest_pos_j).norm();

			if (r > 0.01*horizon)
			{
				Real weight = kernSmooth.Weight(r, horizon);
				Vec3f p = (position[j] - cur_pos_i);
				Real ratio_ij = p.norm() / r;
				J2_i = (ratio_ij - I1_i)*(ratio_ij - I1_i)*weight;
			}
		}
		if (total_weight > EPSILON)
		{
			J2_i /= total_weight;
			J2_i = sqrt(J2_i);
		}
		else
		{
			J2_i = 0.0f;
		}

		Real D1 = 1 - I1_i;		//positive for compression and negative for stretching

		Real s_J2 = J2_i*mu*bulk_stiffiness[i];
		Real s_D1 = D1*lambda*bulk_stiffiness[i];

		//Drucker-Prager yield criterion
		if (s_J2 > s_A + B*s_D1)
		{
			bulk_stiffiness[i] = 0.0f;
		}
	}

	template<typename TDataType>
	void FractureModule<TDataType>::applyPlasticity()
	{
		int num = this->inPosition()->getElementCount();
		uint pDims = cudaGridSize(num, BLOCK_SIZE);

		Real A = this->computeA();
		Real B = this->computeB();

		PM_ComputeInvariants<< <pDims, BLOCK_SIZE >> > (
			this->mBulkStiffness,
			this->inPosition()->getData(),
			this->inRestShape()->getData(),
			this->inHorizon()->getData(),
			A,
			B,
			this->varMu()->getData(),
			this->varLambda()->getData());
		cuSynchronize();
	}

	DEFINE_CLASS(FractureModule);
}