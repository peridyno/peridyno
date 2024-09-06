#include "SurfaceTension.h"
#include "Kernel.h"

namespace dyno
{
	template<typename Real, typename Coord>
	__global__ void ST_ComputeSurfaceEnergy(
		DArray<Real> energyArr,
		DArray<Coord> posArr,
		DArrayList<int> neighbors,
		Real smoothingLength)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.size()) return;

		Real total_weight = Real(0);
		Coord dir_i(0);

		SmoothKernel<Real> kern;

		Coord pos_i = posArr[pId];
		List<int>& nbrIds_i = neighbors[pId];
		int nbSize = nbrIds_i.size();
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = nbrIds_i[ne];
			Real r = (pos_i - posArr[j]).norm();

			if (r > EPSILON)
			{
				Real weight = -kern.Gradient(r, smoothingLength);
				total_weight += weight;
				dir_i += (posArr[j] - pos_i)*(weight / r);
			}
		}

		total_weight = total_weight < EPSILON ? 1.0f : total_weight;
		Real absDir = dir_i.norm() / total_weight;

		energyArr[pId] = absDir*absDir;
	}

	template<typename Real, typename Coord>
	__global__ void ST_ComputeSurfaceTension(
		DArray<Coord> velArr, 
		DArray<Real> energyArr, 
		DArray<Coord> posArr, 
		DArrayList<int> neighbors,
		Real smoothingLength,
		Real mass,
		Real restDensity,
		float dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.size()) return;

		Real Vref = mass / restDensity;

		float alpha = (float) 945.0f / (32.0f * (float)M_PI * smoothingLength * smoothingLength * smoothingLength);
		float ceof = 16000.0f * alpha;

		SmoothKernel<Real> kern;

		Coord F_i(0);
		Coord dv_pi(0);
		Coord pos_i = posArr[pId];
		List<int>& nbrIds_i = neighbors[pId];
		int nbSize = nbrIds_i.size();
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = nbrIds_i[ne];
			float r = (pos_i - posArr[j]).norm();

			if (r > EPSILON)
			{
				Coord temp = Vref*Vref*kern.Gradient(r, smoothingLength)*(posArr[j] - pos_i) * (1.0f / r);
				Coord dv_ij = dt * ceof*1.0f*(energyArr[pId])*temp / mass;
				F_i += dv_ij;
			}
		}
		velArr[pId] -= F_i;
	}

	template<typename TDataType>
	SurfaceTension<TDataType>::SurfaceTension()
		: ComputeModule()
		, m_intensity(Real(1))
		, m_soothingLength(Real(0.0125))
	{

	}

	template<typename TDataType>
	void SurfaceTension<TDataType>::compute()
	{
	}

	DEFINE_CLASS(SurfaceTension);
}