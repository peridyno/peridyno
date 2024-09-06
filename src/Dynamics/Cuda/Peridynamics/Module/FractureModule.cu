#include "FractureModule.h"

#include "ParticleSystem/Module/SummationDensity.h"
#include "ParticleSystem/Module/Kernel.h"

namespace dyno
{
	template<typename TDataType>
	FractureModule<TDataType>::FractureModule()
		: ElastoplasticityModule<TDataType>()
	{
		this->setCohesion(0.001);
	}

	template <typename Real, typename Coord, typename Bond>
	__global__ void PM_ComputeInvariants(
		DArray<Real> bulk_stiffiness,
		DArray<Coord> X,
		DArray<Coord> Y,
		DArrayList<Bond> bonds,
		Real horizon,
		Real A,
		Real B,
		Real mu,
		Real lambda)
	{
		int i = threadIdx.x + (blockIdx.x * blockDim.x);
		if (i >= Y.size()) return;

		CorrectedKernel<Real> kernSmooth;

		Real s_A = A;

		List<Bond>& bonds_i = bonds[i];
		Coord x_i = X[i];
		Coord y_i = Y[i];

		Real I1_i = 0.0f;
		Real J2_i = 0.0f;

		//compute the first and second invariants of the deformation state, i.e., I1 and J2
		int size_i = bonds_i.size();
		Real total_weight = Real(0);
		for (int ne = 1; ne < size_i; ne++)
		{
			Bond bond_ij = bonds_i[ne];
			int j = bond_ij.idx;
			Coord x_j = X[j];
			
			Real r = (x_i - x_j).norm();

			if (r > 0.01*horizon)
			{
				Real weight = kernSmooth.Weight(r, horizon);
				Coord p = (Y[j] - y_i);
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

		for (int ne = 0; ne < size_i; ne++)
		{
			Bond bond_ij = bonds_i[ne];
			int j = bond_ij.idx;
			Coord x_j = X[j];
			Real r = (x_i - x_j).norm();

			if (r > 0.01*horizon)
			{
				Real weight = kernSmooth.Weight(r, horizon);
				Vec3f p = (Y[j] - y_i);
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
		int num = this->inY()->size();
		uint pDims = cudaGridSize(num, BLOCK_SIZE);

		Real A = this->computeA();
		Real B = this->computeB();

		PM_ComputeInvariants<< <pDims, BLOCK_SIZE >> > (
			this->mBulkStiffness,
			this->inX()->getData(),
			this->inY()->getData(),
			this->inBonds()->getData(),
			this->inHorizon()->getData(),
			A,
			B,
			this->varMu()->getData(),
			this->varLambda()->getData());
		cuSynchronize();
	}

	DEFINE_CLASS(FractureModule);
}