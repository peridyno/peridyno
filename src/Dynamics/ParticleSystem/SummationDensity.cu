#include <cuda_runtime.h>
#include "SummationDensity.h"
#include "Framework/MechanicalState.h"
#include "Framework/Node.h"
#include "Kernel.h"

namespace dyno
{
	IMPLEMENT_CLASS_1(SummationDensity, TDataType)

	template<typename Real, typename Coord>
	__global__ void K_ComputeDensity(
		DArray<Real> rhoArr,
		DArray<Coord> posArr,
		NeighborList<int> neighbors,
		Real smoothingLength,
		Real mass
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.size()) return;

		SpikyKernel<Real> kern;
		Real r;
		Real rho_i = Real(0);
		Coord pos_i = posArr[pId];
		int nbSize = neighbors.getNeighborSize(pId);
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = neighbors.getElement(pId, ne);
			r = (pos_i - posArr[j]).norm();
			rho_i += mass*kern.Weight(r, smoothingLength);
		}
		rhoArr[pId] = rho_i;
	}

	template<typename TDataType>
	SummationDensity<TDataType>::SummationDensity()
		: ComputeModule()
		, m_factor(Real(1))
	{
		this->varRestDensity()->setValue(Real(1000));
		this->varSmoothingLength()->setValue(Real(0.011));
		this->varSamplingDistance()->setValue(Real(0.005));

		std::function<void()> callback = std::bind(&SummationDensity<TDataType>::calculateScalingFactor, this);

		this->varRestDensity()->setCallBackFunc(callback);
		this->varSmoothingLength()->setCallBackFunc(callback);
		this->varSamplingDistance()->setCallBackFunc(callback);

		//Should be called after above four parameters are all set, this function will recalculate m_factor
		calculateParticleMass();
		calculateScalingFactor();
	}

	template<typename TDataType>
	void SummationDensity<TDataType>::compute()
	{
		int p_num = this->inPosition()->getElementCount();
		int n_num = this->inNeighborIndex()->getElementCount();
		if (p_num != n_num)
		{
			Log::sendMessage(Log::Error, "The input array sizes of DensitySummation are not compatible!");
			return;
		}

		if (this->outDensity()->getElementCount() != p_num)
		{
			this->outDensity()->setElementCount(p_num);
		}

		compute(
			this->outDensity()->getValue(),
			this->inPosition()->getValue(),
			this->inNeighborIndex()->getValue(),
			this->varSmoothingLength()->getValue(),
			m_particle_mass);

		this->outDensity()->tagModified(true);
	}


	template<typename TDataType>
	void SummationDensity<TDataType>::compute(DArray<Real>& rho)
	{
		compute(
			rho,
			this->inPosition()->getValue(),
			this->inNeighborIndex()->getValue(),
			this->varSmoothingLength()->getValue(),
			m_particle_mass);
	}

	template<typename TDataType>
	void SummationDensity<TDataType>::compute(
		DArray<Real>& rho, 
		DArray<Coord>& pos,
		NeighborList<int>& neighbors, 
		Real smoothingLength,
		Real mass)
	{
		cuExecute(rho.size(), K_ComputeDensity,
			rho, 
			pos, 
			neighbors, 
			smoothingLength, 
			m_factor*mass);

		
	}

	template<typename TDataType>
	void SummationDensity<TDataType>::calculateScalingFactor()
	{
		Real d = this->varSamplingDistance()->getValue();
		Real H = this->varSmoothingLength()->getValue();
		Real rho_0 = this->varRestDensity()->getValue();
		
		Real V = d * d*d;

		SpikyKernel<Real> kern;

		Real total_weight(0);
		int half_res = H / d + 1;
		for (int i = -half_res; i <= half_res; i++)
			for (int j = -half_res; j <= half_res; j++)
				for (int k = -half_res; k <= half_res; k++)
				{
					Real x = i * d;
					Real y = j * d;
					Real z = k * d;
					Real r = sqrt(x * x + y * y + z * z);
					total_weight += V * kern.Weight(r, H);
				}

		m_factor = 1.0 / total_weight;
		m_particle_mass = rho_0 * V;
	}


	template<typename TDataType>
	void SummationDensity<TDataType>::calculateParticleMass()
	{
		Real rho_0 = this->varRestDensity()->getValue();
		Real d = this->varSamplingDistance()->getValue();

		m_particle_mass = d*d*d*rho_0;
	}

}