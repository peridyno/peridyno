#include "SummationDensity.h"

namespace dyno
{
	template<typename TDataType>
	SummationDensity<TDataType>::SummationDensity()
		: ParticleApproximation<TDataType>()
		, m_factor(Real(1))
	{
		this->varRestDensity()->setValue(Real(1000));

		auto callback = std::make_shared<FCallBackFunc>(
			std::bind(&SummationDensity<TDataType>::calculateParticleMass, this));

		this->varRestDensity()->attach(callback);
		this->inSamplingDistance()->attach(callback);

		this->inOther()->tagOptional(true);
		//calculateParticleMass();
	}

	template<typename TDataType>
	void SummationDensity<TDataType>::compute()
	{
		int p_num = this->inPosition()->getDataPtr()->size();
		int n_num = this->inNeighborIds()->getDataPtr()->size();
		if (p_num != n_num) {
			Log::sendMessage(Log::Error, "The input array sizes of DensitySummation are not compatible!");
			return;
		}

		if (this->outDensity()->size() != p_num) {
			this->outDensity()->resize(p_num);
		}

		if (this->inOther()->isEmpty()) {
			compute(
				this->outDensity()->getData(),
				this->inPosition()->getData(),
				this->inNeighborIds()->getData(),
				this->inSmoothingLength()->getData(),
				m_particle_mass);
		}
		else {
			compute(
				this->outDensity()->getData(),
				this->inPosition()->getData(),
				this->inOther()->getData(),
				this->inNeighborIds()->getData(),
				this->inSmoothingLength()->getData(),
				m_particle_mass);
		}
	}

	template<typename Real, typename Coord, typename Kernel>
	__global__ void SD_ComputeDensity(
		DArray<Real> rhoArr,
		DArray<Coord> posArr,
		DArrayList<int> neighbors,
		Real smoothingLength,
		Real mass,
		Kernel weight,
		Real scale)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.size()) return;

		Real r;
		Real rho_i = Real(0);
		Coord pos_i = posArr[pId];
		List<int>& list_i = neighbors[pId];
		int nbSize = list_i.size();
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = list_i[ne];
			r = (pos_i - posArr[j]).norm();
			rho_i += mass * weight(r, smoothingLength, scale);
		}

		rhoArr[pId] = rho_i;
	}

	template<typename Real, typename Coord, typename Kernel>
	__global__ void SD_ComputeDensity(
		DArray<Real> rhoArr,
		DArray<Coord> posArr,
		DArray<Coord> posQueried,
		DArrayList<int> neighbors,
		Real smoothingLength,
		Real mass,
		Kernel weight,
		Real scale)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.size()) return;

		Real r;
		Real rho_i = Real(0);
		Coord pos_i = posArr[pId];
		List<int>& list_i = neighbors[pId];
		int nbSize = list_i.size();
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = list_i[ne];
			r = (pos_i - posQueried[j]).norm();
			rho_i += mass * weight(r, smoothingLength, scale);
		}

		rhoArr[pId] = rho_i;
	}

	template<typename TDataType>
	void SummationDensity<TDataType>::compute(
		DArray<Real>& rho, 
		DArray<Coord>& pos,
		DArrayList<int>& neighbors,
		Real smoothingLength,
		Real mass)
	{
		cuZerothOrder(rho.size(), this->varKernelType()->getDataPtr()->currentKey(), mScalingFactor,
			SD_ComputeDensity,
			rho,
			pos,
			neighbors,
			smoothingLength,
			mass);
	}

	template<typename TDataType>
	void SummationDensity<TDataType>::compute(DArray<Real>& rho, DArray<Coord>& pos, DArray<Coord>& posQueried, DArrayList<int>& neighbors, Real smoothingLength, Real mass)
	{
		cuZerothOrder(rho.size(), this->varKernelType()->getDataPtr()->currentKey(), mScalingFactor,
			SD_ComputeDensity,
			rho,
			pos,
			posQueried,
			neighbors,
			smoothingLength,
			mass);
	}

	template<typename TDataType>
	void SummationDensity<TDataType>::calculateParticleMass()
	{
		Real rho_0 = this->varRestDensity()->getData();
		Real d = this->inSamplingDistance()->getData();

		m_particle_mass = d * d*d*rho_0;
	}

	DEFINE_CLASS(SummationDensity);
}