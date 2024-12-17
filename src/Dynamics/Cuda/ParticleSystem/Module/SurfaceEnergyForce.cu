#include "SurfaceEnergyForce.h"

namespace dyno
{
	template<typename TDataType>
	SurfaceEnergyForce<TDataType>::SurfaceEnergyForce()
		: ParticleApproximation<TDataType>()
	{
		this->varKernelType()->setCurrentKey(EKernelType::KT_Smooth);
	}

	template<typename TDataType>
	SurfaceEnergyForce<TDataType>::~SurfaceEnergyForce()
	{
		mFreeSurfaceEnergy.clear();
	}

	//Equation 3
	template<typename Real, typename Coord, typename Kernel>
	__global__ void ST_ComputeSurfaceEnergy(
		DArray<Real> energyArr,
		DArray<Coord> posArr,
		DArrayList<int> neighbors,
		Real smoothingLength,
		Real samplingDistance,
		Kernel gradient,
		Real scale)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.size()) return;

		Real V_0 = samplingDistance * samplingDistance * samplingDistance;

		Real total_weight = Real(0);
		Coord dir_i(0);

		Coord pos_i = posArr[pId];
		List<int>& nbrIds_i = neighbors[pId];
		int nbSize = nbrIds_i.size();
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = nbrIds_i[ne];
			Real r = (pos_i - posArr[j]).norm();

			if (r > EPSILON)
			{
				Real weight = -V_0 * gradient(r, smoothingLength, scale);
				total_weight += weight;
				dir_i += (posArr[j] - pos_i) * (weight / r);
			}
		}

		total_weight = total_weight < EPSILON ? 1.0f : total_weight;
		Real absDir = dir_i.norm() / total_weight;

		energyArr[pId] = absDir * absDir;
	}

	//Equation 4
	template<typename Real, typename Coord, typename Kernel>
	__global__ void ST_ComputeSurfaceTension(
		DArray<Coord> velArr,
		DArray<Real> energyArr,
		DArray<Coord> posArr,
		DArrayList<int> neighbors,
		Real smoothingLength,
		Real samplingDistance,
		Real kappa,
		Real density,
		Real dt,
		Kernel gradient,
		Real scale)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.size()) return;

		Real V_0 = samplingDistance * samplingDistance * samplingDistance;

		//A hack in using 1000000, to be refined in the future
		Real ceof = 1000000 * kappa;

		Coord F_i(0);
		Coord dv_pi(0);
		Coord pos_i = posArr[pId];
		List<int>& nbrIds_i = neighbors[pId];
		int nbSize = nbrIds_i.size();
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = nbrIds_i[ne];
			Real r = (pos_i - posArr[j]).norm();

			if (r > EPSILON)
			{
				Coord F_ij = V_0 * V_0 * gradient(r, smoothingLength, scale) * (posArr[j] - pos_i) * (1.0f / r);
				F_i += F_ij;
			}
		}

		velArr[pId] -= dt * ceof * F_i / (density * V_0);
	}

	template<typename TDataType>
	void SurfaceEnergyForce<TDataType>::compute()
	{
		Real kappa = this->varKappa()->getValue();
		Real rho = this->varRestDensity()->getValue();

		auto dt = this->inTimeStep()->getValue();

		auto& pos = this->inPosition()->constData();
		auto& vel = this->inVelocity()->getData();
		auto& neighbors = this->inNeighborIds()->constData();

		auto h = this->inSmoothingLength()->getValue();
		auto d = this->inSamplingDistance()->getValue();

		int num = pos.size();

		if (num != mFreeSurfaceEnergy.size())
			mFreeSurfaceEnergy.resize(num);

		cuFirstOrder(num, this->varKernelType()->getDataPtr()->currentKey(), this->mScalingFactor,
			ST_ComputeSurfaceEnergy,
			mFreeSurfaceEnergy,
			pos,
			neighbors,
			h,
			d);

		cuFirstOrder(num, this->varKernelType()->getDataPtr()->currentKey(), this->mScalingFactor,
			ST_ComputeSurfaceTension,
			vel,
			mFreeSurfaceEnergy,
			pos,
			neighbors,
			h,
			d,
			kappa,
			rho,
			dt);
	}

	DEFINE_CLASS(SurfaceEnergyForce);
}