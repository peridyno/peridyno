#include "GranularModule.h"
#include "Particlesystem/SummationDensity.h"

namespace dyno
{
	template<typename TDataType>
	GranularModule<TDataType>::GranularModule()
		: ElastoplasticityModule<TDataType>()
	{
	}


// 	template<typename TDataType>
// 	bool GranularModule<TDataType>::initializeImpl()
// 	{
// 		//TODO: replace
// 		m_densitySum = std::make_shared<SummationDensity<TDataType>>();
// 
// 		this->inHorizon()->connect(m_densitySum->inSmoothingLength());
// 		this->inPosition()->connect(m_densitySum->inPosition());
// 		this->inNeighborIds()->connect(m_densitySum->inNeighborIds());
// 
// 		m_densitySum->initialize();
// 
// 		return ElastoplasticityModule<TDataType>::initializeImpl();
// 	}


	__device__ Real Hardening(Real rho, Real restRho)
	{
		if (rho >= restRho)
		{
			float ratio = rho / restRho;
			//ratio = ratio > 1.1f ? 1.1f : ratio;
			return pow(Real(M_E), Real(ratio - 1.0f));
		}
		else
		{
			return Real(0);
		};
	}

	template <typename Real>
	__global__ void PM_ComputeStiffness(
		DArray<Real> stiffiness,
		DArray<Real> density)
	{
		int i = threadIdx.x + (blockIdx.x * blockDim.x);
		if (i >= stiffiness.size()) return;

		stiffiness[i] = Hardening(density[i], Real(1000));
	}

	template<typename TDataType>
	void GranularModule<TDataType>::computeMaterialStiffness()
	{
		int num = this->inPosition()->getElementCount();
		uint pDims = cudaGridSize(num, BLOCK_SIZE);

		m_densitySum->compute();

		PM_ComputeStiffness << <pDims, BLOCK_SIZE >> > (
			this->mBulkStiffness,
			m_densitySum->outDensity()->getData());
		cuSynchronize();
	}

	DEFINE_CLASS(GranularModule);
}