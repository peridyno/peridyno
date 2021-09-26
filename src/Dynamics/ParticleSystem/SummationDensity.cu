#include "SummationDensity.h"
#include "Node.h"
#include "Kernel.h"

namespace dyno
{
	IMPLEMENT_CLASS_1(SummationDensity, TDataType)

	template<typename Real, typename Coord>
	__global__ void SD_ComputeDensity(
		DArray<Real> rhoArr,
		DArray<Coord> posArr,
		DArrayList<int> neighbors,
		Real smoothingLength,
		Real mass)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.size()) return;

		SpikyKernel<Real> kern;
		Real r;
		Real rho_i = Real(0);
		Coord pos_i = posArr[pId];
		List<int>& list_i = neighbors[pId];
		int nbSize = list_i.size();
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = list_i[ne];
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
		this->inSmoothingLength()->setValue(Real(0.011));
		this->inSamplingDistance()->setValue(Real(0.005));

		std::function<void()> callback = std::bind(&SummationDensity<TDataType>::calculateScalingFactor, this);

		this->varRestDensity()->setCallBackFunc(callback);
		this->inSmoothingLength()->setCallBackFunc(callback);
		this->inSamplingDistance()->setCallBackFunc(callback);

		//Should be called after above four parameters are all set, this function will recalculate m_factor
		calculateParticleMass();
		calculateScalingFactor();
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

		if (this->outDensity()->getElementCount() != p_num) {
			this->outDensity()->setElementCount(p_num);
		}

		compute(
			this->outDensity()->getData(),
			this->inPosition()->getData(),
			this->inNeighborIds()->getData(),
			this->inSmoothingLength()->getData(),
			m_particle_mass);
	}


	template<typename TDataType>
	void SummationDensity<TDataType>::compute(DArray<Real>& rho)
	{
		compute(
			rho,
			this->inPosition()->getData(),
			this->inNeighborIds()->getData(),
			this->inSmoothingLength()->getData(),
			m_particle_mass);
	}

	template<typename TDataType>
	void SummationDensity<TDataType>::compute(
		DArray<Real>& rho, 
		DArray<Coord>& pos,
		DArrayList<int>& neighbors,
		Real smoothingLength,
		Real mass)
	{
		cuExecute(rho.size(), 
			SD_ComputeDensity,
			rho, 
			pos, 
			neighbors, 
			smoothingLength, 
			m_factor*mass);
	}

	template<typename TDataType>
	void SummationDensity<TDataType>::calculateScalingFactor()
	{
		Real d = this->inSamplingDistance()->getData();
		Real H = this->inSmoothingLength()->getData();
		Real rho_0 = this->varRestDensity()->getData();
		
		Real V = d * d*d;

		SpikyKernel<Real> kern;

		Real total_weight(0);
		int half_res = (int)(H / d + 1);
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

		m_factor = Real(1) / total_weight;
		m_particle_mass = rho_0 * V;
	}


	template<typename TDataType>
	void SummationDensity<TDataType>::calculateParticleMass()
	{
		Real rho_0 = this->varRestDensity()->getData();
		Real d = this->inSamplingDistance()->getData();

		m_particle_mass = d*d*d*rho_0;
	}

// #ifdef PRECISION_FLOAT
// template class SummationDensity<DataType3f>;
// #else
// template class SummationDensity2<DataType3d>;
// #endif

	DEFINE_CLASS(SummationDensity);
}