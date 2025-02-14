#include "ImplicitISPH.h"

#include "SummationDensity.h"

namespace dyno
{
	template<typename TDataType>
	ImplicitISPH<TDataType>::ImplicitISPH()
		: ParticleApproximation<TDataType>()
	{
		//this->varIterationNumber()->setValue(3);
		this->varKappa()->setValue(200);
		this->varRestDensity()->setValue(Real(1000));

		mSummation = std::make_shared<SummationDensity<TDataType>>();

		this->inSmoothingLength()->connect(mSummation->inSmoothingLength());
		this->inSamplingDistance()->connect(mSummation->inSamplingDistance());
		this->inPosition()->connect(mSummation->inPosition());
		this->inNeighborIds()->connect(mSummation->inNeighborIds());

		mSummation->outDensity()->connect(this->outDensity());
	}

	template<typename TDataType>
	ImplicitISPH<TDataType>::~ImplicitISPH()
	{
		mSourceTerm.clear();
		mDii.clear();
		mAii.clear();
		mAnPn.clear();
		mSumDijPj.clear();
		mPressrue.clear();
		mOldPressrue.clear();
		m_Residual.clear();
		mPredictDensity.clear();
		mDensityAdv.clear();

	}


	/*
	*@Equation: Source_i = rho_0 - (rho_i + dt * div(vi))
	*/
	template <typename Real, typename Coord, typename Kernel>
	__global__ void IISPH_SourceTermCompute(
		DArray<Real> Sources,
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
		if (pId >= Sources.size()) return;

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

		Sources[pId] = rest_density - (rhoArr[pId] + dt * div_i);

		if (Sources[pId] > 0.0f) Sources[pId] = 0.0f;

	}

	/*
	*@Equation: Coord D_ii = - dt^2 * sum_j { mass/(rho_i)^2 * Grad_Wij }
	*/
	template <typename Real, typename Coord, typename Kernel>
	__global__ void IISPH_DiiCoefficientOfPpeCompute(
		DArray<Coord> D,
		DArray<Coord> posArr,
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
		if (pId >= D.size()) return;

		Coord pos_i = posArr[pId];

		Coord sum(0.0f);

		List<int>& list_i = neighbors[pId];
		int nbSize = list_i.size();

		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = list_i[ne];
			Real r = (pos_i - posArr[j]).norm();

			if (r > EPSILON)
			{
				Coord g = gradient(r, smoothingLength, scale) * (pos_i - posArr[j]) * (1.0f / r);
				sum -= mass / (rhoArr[pId] * rhoArr[pId]) * g;
			}
		}

		D[pId] = sum * dt * dt;
	}



	/*
	*@Equation: Coord A_ii = - sum_j { mass (D_ii - D_ji) Grad_Wij };
	*			D_ij = - dt^2 * mass / rho_i^2 * Grad_Wij;
	*/
	template <typename Real, typename Coord, typename Kernel>
	__global__ void IISPH_AiiCoefficientOfPpeCompute(
		DArray<Real> Aii,
		DArray<Coord> Dii,
		DArray<Coord> posArr,
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
		if (pId >= Aii.size()) return;

		Coord pos_i = posArr[pId];

		Real sum = (0.0f);

		List<int>& list_i = neighbors[pId];
		int nbSize = list_i.size();

		Coord dji(0.0f);

		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = list_i[ne];
			Real r = (pos_i - posArr[j]).norm();

			if (r > EPSILON)
			{
				Coord g_ij = gradient(r, smoothingLength, scale) * (pos_i - posArr[j] ) * (1.0f / r);
				Coord g_ji = gradient(r, smoothingLength, scale) * (posArr[j] - pos_i) * (1.0f / r);
				dji = -1.0 * dt * dt * mass  * g_ji / (rhoArr[pId] * rhoArr[pId]);			//grad_W_ij			???(-1)
				sum += mass * g_ij.dot(Dii[pId] - dji);		//grad_W_ji		???
			}
		}

		Aii[pId] = sum;
	}


	/*
	*@Equation: Coord DijPj = D_ij * Pj;
	*			D_ij = - dt^2 * mass * Grad_Wij * Pj / rho_i^2 ;
	*/
	template <typename Real, typename Coord, typename Kernel>
	__global__ void IISPH_Sum_DijDotPjInPPE(
		DArray<Coord> SumDijPj,
		DArray<Real> pressures,
		DArray<Coord> posArr,
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
		if (pId >= SumDijPj.size()) return;

		Coord pos_i = posArr[pId];

		List<int>& list_i = neighbors[pId];
		int nbSize = list_i.size();

		Coord sum(0.0f);

		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = list_i[ne];
			Real r_ij = (pos_i - posArr[j]).norm();
			if (r_ij > EPSILON)
			{
				Coord g_ij = gradient(r_ij, smoothingLength, scale) * (pos_i - posArr[j]) * (1.0f / r_ij);
				sum -= dt * dt * mass * g_ij * pressures[j] / (rhoArr[j] * rhoArr[j]);
			}
		}
		SumDijPj[pId] = sum;
	}

	/*
	*@Equation: Coord AnPn = sum_n {An * Pn} - Aij;
	*			AnPn =  Sum_j { mass * {(Sum_j Dij * Pj) - D_jj * Pj - Sum_(k!=i) D_jk * Pj } Grad_Wij};
	*/
	template <typename Real, typename Coord, typename Kernel>
	__global__ void IISPH_AnDotPnInPPE(
		DArray<Real> AnPn,
		DArray<Real> pressures,
		DArray<Coord> Dii,
		DArray<Coord> SumDijPj,
		DArray<Coord> posArr,
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
		if (pId >= AnPn.size()) return;

		Coord pos_i = posArr[pId];

		List<int>& list_i = neighbors[pId];
		int nbSize = list_i.size();

		Coord d_ij(0.0f);
		Coord d_jk(0.0f);

		Real sum = 0.0f;

		Coord sum_DjkPk(0.0f);

		Coord d_ji(0.0f);

		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = list_i[ne];

			Real r_ij = (pos_i - posArr[j]).norm();

			if (r_ij > EPSILON)
			{
				Coord g_ij = gradient(r_ij, smoothingLength, scale) * (pos_i - posArr[j]) * (1.0f / r_ij);

				//Coord g_ji = gradient(r_ij, smoothingLength, scale) * (posArr[j] - pos_i) * (1.0f / r_ij);

				//Coord d_ji = -dt * dt * mass * g_ji / (rhoArr[pId] * rhoArr[pId]);

				//sum += mass * g_ij.dot(SumDijPj[pId] - Dii[j] * pressures[j] - (SumDijPj[j] - d_ji * pressures[pId] ));

				//Coord sumDjkPk = SumDijPj[j];
				Coord sumDjkPk(0);

				List<int>& list_j = neighbors[j];
				for (int nj = 0; nj < list_j.size(); nj++)
				{
					int k = list_i[nj];
					Real r_jk = (posArr[j] - posArr[k]).norm();
					if (r_jk > EPSILON && k != pId)
					{
						Coord g_jk = gradient(r_jk, smoothingLength, scale) * (posArr[j] - posArr[k]) * (1.0f / r_jk);
						d_jk = (-dt * dt * mass / (rhoArr[k] * rhoArr[k])) * g_jk;

						sumDjkPk += d_jk * pressures[k];
					}
				}

				sum += mass * g_ij.dot(SumDijPj[pId] - Dii[j] * pressures[j] - sumDjkPk);
			}

		}
		AnPn[pId] = sum;
	}

	/*
	*@Brief:	Relaxed Jacobi Method. 
	*@Equation: pressure_new[i] = Omega * pressure_old[i] + (1 - Omega) * (source[pId] - AnPn) / Aii
	*			
	*/
	template <typename Real>
	__global__ void IISPH_PressureJacobiCompute(
		DArray<Real> pressures,
		DArray<Real> oldPressures,
		DArray<Real> AnPn,
		DArray<Real> Source,
		DArray<Real> Aii,
		DArray<Real> Residual,
		Real omega
	) {

		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= pressures.size()) return;

	
		if (abs(Aii[pId]) > EPSILON)
		{
			pressures[pId] = oldPressures[pId] * (1.0 - omega) + omega * (Source[pId] - AnPn[pId]) / Aii[pId];
		}
		else {
			pressures[pId] = 0.0f;
		}

		/*
		*@note Clamping negative pressures will lead to incorrect estimation of the residual in the PPE.
		*/
		//if (pressures[pId] < 0.0f)
		//	pressures[pId] = 0.0f;

		Residual[pId] = abs(Source[pId] - AnPn[pId] - Aii[pId]* pressures[pId]);


	}


	template<typename TDataType>
	void ImplicitISPH<TDataType>::compute()
	{
		int num = this->inPosition()->size();
	
		if (this->outDensity()->size() != this->inPosition()->size())
			this->outDensity()->resize(this->inPosition()->size());

		if (mPredictDensity.size() != this->inPosition()->size())
			mPredictDensity.resize(this->inPosition()->size());

		if (mDensityAdv.size() != this->inPosition()->size())
			mDensityAdv.resize(this->inPosition()->size());

		if (mPressrue.size() != this->inPosition()->size())
			mPressrue.resize(this->inPosition()->size());

		if (mSourceTerm.size() != this->inPosition()->size())
			mSourceTerm.resize(this->inPosition()->size());

		if (mDii.size() != this->inPosition()->size())
			mDii.resize(this->inPosition()->size());

		if (mAii.size() != this->inPosition()->size())
			mAii.resize(this->inPosition()->size());

		if (mAnPn.size() != this->inPosition()->size())
			mAnPn.resize(this->inPosition()->size());
		
		if (mPressrue.size() != this->inPosition()->size())
			mPressrue.resize(this->inPosition()->size());

		if (mSumDijPj.size() != this->inPosition()->size())
			mSumDijPj.resize(this->inPosition()->size());

		if (m_Residual.size() != this->inPosition()->size())
			m_Residual.resize(this->inPosition()->size());

		if (mOldPressrue.size() != this->inPosition()->size())
			mOldPressrue.resize(this->inPosition()->size());
		

		mSummation->varRestDensity()->setValue(this->varRestDensity()->getValue());
		mSummation->varKernelType()->setCurrentKey(this->varKernelType()->currentKey());
		mSummation->update();

		this->PreIterationCompute();

		int it = 0;

		int itNum = this->varIterationNumber()->getValue();

		mOldPressrue.reset();
		mPressrue.reset();

// 		Real error_max = takeOneIteration();
// 		Real error = error_max;

		while ( (it++ < itNum))
		{
			takeOneIteration();
			//std::cout << "Residual: " << error / error_max * 100.0f << "%" << std::endl;
		}

		updateVelocity();
	}


	template<typename TDataType>
	void ImplicitISPH<TDataType>::PreIterationCompute()
	{
		Real dt = this->inTimeStep()->getData();
		int num = this->inPosition()->size();
		Real rho_0 = this->varRestDensity()->getValue();

		cuFirstOrder(num, this->varKernelType()->getDataPtr()->currentKey(), this->mScalingFactor,
			IISPH_DiiCoefficientOfPpeCompute,
			mDii,
			this->inPosition()->getData(),
			mSummation->outDensity()->getData(),
			this->inNeighborIds()->getData(),
			this->inTimeStep()->getValue(),
			rho_0,
			mSummation->getParticleMass(),
			this->inSmoothingLength()->getValue()
		);
		
		cuFirstOrder(num, this->varKernelType()->getDataPtr()->currentKey(), this->mScalingFactor,
			IISPH_SourceTermCompute,
			mSourceTerm,
			this->inPosition()->getData(),
			this->inVelocity()->getData(),
			mSummation->outDensity()->getData(),
			this->inNeighborIds()->getData(),
			this->inTimeStep()->getValue(),
			rho_0,
			mSummation->getParticleMass(),
			this->inSmoothingLength()->getValue()
		);

		cuFirstOrder(num, this->varKernelType()->getDataPtr()->currentKey(), this->mScalingFactor,
			IISPH_AiiCoefficientOfPpeCompute,
			mAii,
			mDii,
			this->inPosition()->getData(),
			mSummation->outDensity()->getData(),
			this->inNeighborIds()->getData(),
			this->inTimeStep()->getValue(),
			rho_0,
			mSummation->getParticleMass(),
			this->inSmoothingLength()->getValue()
		);


	}


	template<typename TDataType>
	typename TDataType::Real ImplicitISPH<TDataType>::takeOneIteration()
	{
		Real dt = this->inTimeStep()->getData();
		int num = this->inPosition()->size();
		Real rho_0 = this->varRestDensity()->getValue();

		mOldPressrue.assign(mPressrue);
	
// 		cuFirstOrder(num, this->varKernelType()->getDataPtr()->currentKey(), this->mScalingFactor,
// 			IISPH_Sum_DijDotPjInPPE,
// 			mSumDijPj,
// 			mPressrue,
// 			this->inPosition()->getData(),
// 			mSummation->outDensity()->getData(),
// 			this->inNeighborIds()->getData(),
// 			this->inTimeStep()->getValue(),
// 			rho_0,
// 			mSummation->getParticleMass(),
// 			this->inSmoothingLength()->getValue()
// 		);

		cuFirstOrder(num, this->varKernelType()->getDataPtr()->currentKey(), this->mScalingFactor,
			IISPH_AnDotPnInPPE,
			mAnPn,
			mPressrue,
			mDii,
			mSumDijPj,
			this->inPosition()->getData(),
			mSummation->outDensity()->getData(),
			this->inNeighborIds()->getData(),
			this->inTimeStep()->getValue(),
			rho_0,
			mSummation->getParticleMass(),
			this->inSmoothingLength()->getValue()
		);


		cuExecute(num, IISPH_PressureJacobiCompute,
			mPressrue,
			mOldPressrue,
			mAnPn,
			mSourceTerm,
			mAii,
			m_Residual,
			this->varRelaxedOmega()->getValue()
		);

// 		auto m_reduce = Reduction<Real>::Create(num);
// 		Real error = m_reduce->average(m_Residual.begin(), num) / num;
		return 0;

	}


	template <typename Real, typename Coord, typename Kernel>
	__global__ void IISPH_UpdateVelocity(
		DArray<Coord> velocities,
		DArray<Real> pressures,
		DArray<Coord> posArr,
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
		if (pId >= velocities.size()) return;

		Coord Force_i(0.0f);

		List<int>& list_i = neighbors[pId];
		int nbSize = list_i.size();

		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = list_i[ne];
			Real r_ij = (posArr[pId] - posArr[j]).norm();
			if (r_ij > EPSILON)
			{
				Coord g_ij = gradient(r_ij, smoothingLength, scale) * (posArr[pId] - posArr[j]) * (1.0f / r_ij);
				Force_i -= mass * mass * (pressures[pId] / (rhoArr[pId] * rhoArr[pId]) + pressures[j] / (rhoArr[j] * rhoArr[j])) * g_ij;
			}
		}

		velocities[pId] += dt * Force_i / mass;
	}

	template<typename TDataType>
	void ImplicitISPH<TDataType>::updateVelocity()
	{
		int num = this->inPosition()->size();
		Real dt = this->inTimeStep()->getData();
		Real rho_0 = this->varRestDensity()->getValue();

		cuFirstOrder(num, this->varKernelType()->getDataPtr()->currentKey(), this->mScalingFactor,
			IISPH_UpdateVelocity,
			this->inVelocity()->getData(),
			mPressrue,
			this->inPosition()->getData(),
			mSummation->outDensity()->getData(),
			this->inNeighborIds()->getData(),
			this->inTimeStep()->getValue(),
			rho_0,
			mSummation->getParticleMass(),
			this->inSmoothingLength()->getValue()
		);

		
	}

	DEFINE_CLASS(ImplicitISPH);
}