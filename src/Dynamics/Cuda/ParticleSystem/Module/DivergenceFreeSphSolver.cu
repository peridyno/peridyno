#include "DivergenceFreeSphSolver.h"

#include "SummationDensity.h"

namespace dyno
{
	template<typename TDataType>
	DivergenceFreeSphSolver<TDataType>::DivergenceFreeSphSolver()
		: ParticleApproximation<TDataType>()
	{
		this->varRestDensity()->setValue(Real(1000));

		mSummation = std::make_shared<SummationDensity<TDataType>>();

		this->inSmoothingLength()->connect(mSummation->inSmoothingLength());
		this->inSamplingDistance()->connect(mSummation->inSamplingDistance());
		this->inPosition()->connect(mSummation->inPosition());
		this->inNeighborIds()->connect(mSummation->inNeighborIds());

		mSummation->outDensity()->connect(this->outDensity());
	}

	template<typename TDataType>
	DivergenceFreeSphSolver<TDataType>::~DivergenceFreeSphSolver()
	{
		mKappa_r.clear();
		mKappa_v.clear();
		mAlpha.clear();
		mPredictDensity.clear();
		mDivergence.clear();
	}



	template <typename Real, typename Coord, typename Kernel>
	__global__ void DFSPH_AlphaCompute(
		DArray<Real> alpha,
		DArray<Coord> posArr,
		DArray<Real> rhoArr,
		DArrayList<int> neighbors,
		Real rest_density,
		Real mass,
		Real smoothingLength,
		Kernel gradient,
		Real scale
	) {
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= alpha.size()) return;

		Coord pos_i = posArr[pId];

		Real inv_alpha_i = Real(0);
		Coord grad_ci(0);

		List<int>& list_i = neighbors[pId];
		int nbSize = list_i.size();

		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = list_i[ne];
			Real r = (pos_i - posArr[j]).norm();

			if (r > EPSILON)
			{
				Coord g = mass * gradient(r, smoothingLength, scale) * (pos_i - posArr[j]) * (1.0f / r);
				grad_ci += g;
				inv_alpha_i += g.dot(g);
			}
		}

		inv_alpha_i += grad_ci.dot(grad_ci);

		Real rho_i = rhoArr[pId];

		if (inv_alpha_i < EPSILON) inv_alpha_i = EPSILON;

		alpha[pId] = rho_i / (inv_alpha_i);

		if ((nbSize < 5))	alpha[pId] = 0.0;

		//if (alpha[pId] > 1.0)
		//{
		//	printf("a %f, inv_a %e, rho %f, nb %d \r\n", alpha[pId], inv_alpha_i, rho_i, nbSize);
		//}
	}


	template <typename Real, typename Coord, typename Kernel>
	__global__ void DFSPH_DensityPredict(
		DArray<Real> p_rhoArr,
		DArray<Coord> posArr,
		DArray<Coord> veloArr,
		DArray<Real> rhoArr,
		DArrayList<int> neighbors,
		Real dt,
		Real rest_density,
		Real mass,
		Real smoothingLength,
		Kernel gradient,
		Real scale
	) {

		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= p_rhoArr.size()) return;

		Coord pos_i = posArr[pId];

		Real div_i = 0.0f;

		List<int>& list_i = neighbors[pId];
		int nbSize = list_i.size();

		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = list_i[ne];
			Real r = (pos_i - posArr[j]).norm();

			if (r > EPSILON)
			{
				Coord g = gradient(r, smoothingLength, scale) * (pos_i - posArr[j]) * (1.0f / r);
				div_i += mass * g.dot(veloArr[pId] - veloArr[j]);
			}
		}
		Real rho_i = rhoArr[pId] > rest_density ? rhoArr[pId] : rest_density;

		if (nbSize < 10) div_i = 0;

		p_rhoArr[pId] = rho_i + dt * div_i;

		if (p_rhoArr[pId] < rest_density) p_rhoArr[pId] = rest_density;

	}

	template <typename Real, typename Coord, typename Kernel>
	__global__ void DFSPH_DivergencePredict(
		DArray<Real> divergenceArr,
		DArray<Coord> posArr,
		DArray<Coord> veloArr,
		DArray<Real> rhoArr,
		DArrayList<int> neighbors,
		Real dt,
		Real rest_density,
		Real mass,
		Real smoothingLength,
		Kernel gradient,
		Real scale
	) {

		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= divergenceArr.size()) return;

		Coord pos_i = posArr[pId];

		Real div_i = 0.0f;

		List<int>& list_i = neighbors[pId];
		int nbSize = list_i.size();

		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = list_i[ne];
			Real r = (pos_i - posArr[j]).norm();

			if (r > 100 * EPSILON)
			{
				Coord g = gradient(r, smoothingLength, scale) * (pos_i - posArr[j]) * (1.0f / r);
				div_i += mass * g.dot(veloArr[pId] - veloArr[j]);
			}
		}

		if ((div_i < 0) || (nbSize < 10))	div_i = 0;

		divergenceArr[pId] = div_i;
	}


	template <typename Real>
	__global__ void DFSPH_DensityKappa(
		DArray<Real> KappaArr,
		DArray<Real> alpahArr,
		DArray<Real> predict_RhoArr,
		DArray<Real> rhoArr,
		Real dt,
		Real rho_0
	) {
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= KappaArr.size()) return;
		KappaArr[pId] = alpahArr[pId] * (predict_RhoArr[pId] - rho_0) / (dt * dt);
	}



	template <typename Real>
	__global__ void DFSPH_DivergenceKappa(
		DArray<Real> KappaArr,
		DArray<Real> alpahArr,
		DArray<Real> divergence,
		DArray<Real> rhoArr,
		Real dt,
		Real rho_0
	) {
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= KappaArr.size()) return;
		
		KappaArr[pId] = 1.0 * alpahArr[pId] * divergence[pId] / (dt);
	}

	template <typename Real, typename Coord, typename Kernel>
	__global__ void DFSPH_VelocityUpdatedByKappa(
		DArray<Coord> veloArr,
		DArray<Real> KappaArr,
		DArray<Real> alpahArr,
		DArray<Coord> posArr,
		DArray<Real> predict_RhoArr,
		DArray<Real> rhoArr,
		DArrayList<int> neighbors,
		Real dt,
		Real rest_density,
		Real mass,
		Real smoothingLength,
		Kernel gradient,
		Real scale
	) {
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= veloArr.size()) return;

		Coord pos_i = posArr[pId];

		List<int>& list_i = neighbors[pId];
		int nbSize = list_i.size();

		Coord d_velo(0.0f);

		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = list_i[ne];
			Real r = (pos_i - posArr[j]).norm();
			if (r > EPSILON)
			{
				Coord g = gradient(r, smoothingLength, scale) * (pos_i - posArr[j]) * (1.0f / r);
				{
					//d_velo += mass * (KappaArr[pId] / rhoArr[pId] + KappaArr[j] / rhoArr[j]) * g;
					d_velo += mass * (KappaArr[pId] / (rhoArr[pId] * rhoArr[pId]) + KappaArr[j] / (rhoArr[j] * rhoArr[j]) ) * g;
				}
			}
		}
		veloArr[pId] -= dt * d_velo * rhoArr[pId];

	}

	template<typename TDataType>
	void DivergenceFreeSphSolver<TDataType>::compute()
	{
		int num = this->inPosition()->size();

		if (this->outDensity()->size() != this->inPosition()->size())
			this->outDensity()->resize(this->inPosition()->size());

		if (mKappa_r.size() != this->inPosition()->size())
			mKappa_r.resize(this->inPosition()->size());
		
		if (mKappa_v.size() != this->inPosition()->size())
			mKappa_v.resize(this->inPosition()->size());

		if (mAlpha.size() != this->inPosition()->size())
			mAlpha.resize(this->inPosition()->size());

		if (mPredictDensity.size() != this->inPosition()->size())
			mPredictDensity.resize(this->inPosition()->size());

		if (mDivergence.size() != this->inPosition()->size())
			mDivergence.resize(this->inPosition()->size());

		mSummation->varRestDensity()->setValue(this->varRestDensity()->getValue());
		mSummation->varKernelType()->setCurrentKey(this->varKernelType()->currentKey());
		mSummation->update();

		int MaxItNum = this->varMaxIterationNumber()->getValue();

		this->computeAlpha();
		
		int it = 0;
		if (this->varDivergenceSolverDisabled()->getValue() == false)
		{
			Real v_err = 10000.0f;
			while ((v_err > this->varDivergenceErrorThreshold()->getValue()) && (it < MaxItNum))
			{
				it++;
				v_err = abs(this->takeOneDivergenIteration()) / this->varRestDensity()->getValue();
				std::cout << "Divergence Error: " << v_err << std::endl;
			}
		}

		it = 0;
		if (this->varDensitySolverDisabled()->getValue() == false)
		{
			Real d_err = 10000.0f;
			while ((d_err > this->varDensityErrorThreshold()->getValue()) && (it < MaxItNum))
			{
				it++;
				d_err = abs(this->takeOneDensityIteration());
				std::cout << "Density Error: " << d_err << std::endl;
			}
		}
	}

	template<typename TDataType>
	void DivergenceFreeSphSolver<TDataType>::computeAlpha()
	{

		int num = this->inPosition()->size();
		Real rho_0 = this->varRestDensity()->getValue();

		cuFirstOrder(num, this->varKernelType()->getDataPtr()->currentKey(), this->mScalingFactor,
			DFSPH_AlphaCompute,
			mAlpha,
			this->inPosition()->getData(),
			mSummation->outDensity()->getData(),
			this->inNeighborIds()->getData(),
			rho_0,
			mSummation->getParticleMass(),
			this->inSmoothingLength()->getValue()
		);
	}

	template<typename TDataType>
	TDataType::Real DivergenceFreeSphSolver<TDataType>::takeOneDensityIteration()
	{
		Real dt = this->inTimeStep()->getData();
		int num = this->inPosition()->size();
		Real rho_0 = this->varRestDensity()->getValue();

		cuFirstOrder(num, this->varKernelType()->getDataPtr()->currentKey(), this->mScalingFactor,
			DFSPH_DensityPredict,
			mPredictDensity,
			this->inPosition()->getData(),
			this->inVelocity()->getData(),
			mSummation->outDensity()->getData(),
			this->inNeighborIds()->getData(),
			this->inTimeStep()->getValue(),
			rho_0,
			mSummation->getParticleMass(),
			this->inSmoothingLength()->getValue()
		);

		cuExecute(num, DFSPH_DensityKappa,
			mKappa_r,
			mAlpha,
			mPredictDensity,
			mSummation->outDensity()->getData(),
			this->inTimeStep()->getValue(),
			rho_0
		);

		cuFirstOrder(num, this->varKernelType()->getDataPtr()->currentKey(), this->mScalingFactor,
			DFSPH_VelocityUpdatedByKappa,
			this->inVelocity()->getData(),
			mKappa_r,
			mAlpha,
			this->inPosition()->getData(),
			mPredictDensity,
			mSummation->outDensity()->getData(),
			this->inNeighborIds()->getData(),
			this->inTimeStep()->getValue(),
			rho_0,
			mSummation->getParticleMass(),
			this->inSmoothingLength()->getValue()
		);

	   auto m_reduce = Reduction<Real>::Create(num);
	   Real avr_pdensity = m_reduce->average(mPredictDensity.begin(), num);
	   Real err = (avr_pdensity - rho_0) / rho_0;
	   delete m_reduce;
	   return err;
	}

	template<typename TDataType>
	TDataType::Real DivergenceFreeSphSolver<TDataType>::takeOneDivergenIteration()
	{
		Real err = 0.0f;
	
		Real dt = this->inTimeStep()->getData();
		int num = this->inPosition()->size();
		Real rho_0 = this->varRestDensity()->getValue();

		cuFirstOrder(num, this->varKernelType()->getDataPtr()->currentKey(), this->mScalingFactor,
			DFSPH_DivergencePredict,
			mDivergence,
			this->inPosition()->getData(),
			this->inVelocity()->getData(),
			mSummation->outDensity()->getData(),
			this->inNeighborIds()->getData(),
			this->inTimeStep()->getValue(),
			rho_0,
			mSummation->getParticleMass(),
			this->inSmoothingLength()->getValue()
		);

		cuExecute(num, DFSPH_DivergenceKappa,
			mKappa_v,
			mAlpha,
			mDivergence,
			mSummation->outDensity()->getData(),
			this->inTimeStep()->getValue(),
			rho_0
		);

		cuFirstOrder(num, this->varKernelType()->getDataPtr()->currentKey(), this->mScalingFactor,
			DFSPH_VelocityUpdatedByKappa,
			this->inVelocity()->getData(),
			mKappa_v,
			mAlpha,
			this->inPosition()->getData(),
			mPredictDensity,
			mSummation->outDensity()->getData(),
			this->inNeighborIds()->getData(),
			this->inTimeStep()->getValue(),
			rho_0,
			mSummation->getParticleMass(),
			this->inSmoothingLength()->getValue()
		);

		auto m_reduce2 = Reduction<Real>::Create(num);
		Real avr_div = m_reduce2->average(mDivergence.begin(), num);
		err = avr_div;
		delete m_reduce2;
		return err;
	}

	DEFINE_CLASS(DivergenceFreeSphSolver);
}