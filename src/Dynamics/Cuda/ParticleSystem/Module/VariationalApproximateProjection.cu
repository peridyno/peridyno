#include "VariationalApproximateProjection.h"
#include "SummationDensity.h"
#include "Algorithm/Function2Pt.h"
#include "Algorithm/Functional.h"
#include "Algorithm/Arithmetic.h"
#include "Algorithm/Reduction.h"


namespace dyno
{
	template<typename TDataType>
	VariationalApproximateProjection<TDataType>::VariationalApproximateProjection()
		: ParticleApproximation<TDataType>()
		, mAirPressure(Real(0))
		, mAlphaMax(Real(0))
		, mAMax(Real(0))
		, m_reduce(NULL)
		, m_arithmetic(NULL)
	{
		mDensityCalculator = std::make_shared<SummationDensity<TDataType>>();

		this->varRestDensity()->connect(mDensityCalculator->varRestDensity());
		this->inSmoothingLength()->connect(mDensityCalculator->inSmoothingLength());
		this->inSamplingDistance()->connect(mDensityCalculator->inSamplingDistance());

		this->inPosition()->connect(mDensityCalculator->inPosition());
		this->inNeighborIds()->connect(mDensityCalculator->inNeighborIds());

		this->inAttribute()->tagOptional(true);
		this->inNormal()->tagOptional(true);

		this->varKernelType()->getDataPtr()->setCurrentKey(8);

		auto callback = std::make_shared<FCallBackFunc>(std::bind(&VariationalApproximateProjection<TDataType>::varChanged, this));
		this->varKernelType()->attach(callback);
		this->varRestDensity()->attach(callback);
	}

	template<typename TDataType>
	VariationalApproximateProjection<TDataType>::~VariationalApproximateProjection()
	{
		mAlpha.clear();
		mAii.clear();
		mAiiFluid.clear();
		mAiiTotal.clear();
		mPressure.clear();
		mDivergence.clear();
		mIsSurface.clear();

		m_y.clear();
		m_r.clear();
		m_p.clear();

		mPressure.clear();

		if (m_reduce)
		{
			delete m_reduce;
		}
		if (m_arithmetic)
		{
			delete m_arithmetic;
		}
	}

	template <typename Real, typename Coord, typename Kernel>
	__global__ void VAP_ComputeAlpha
	(
		DArray<Real> alpha,
		DArray<Coord> position,
		DArray<Attribute> attribute,
		DArrayList<int> neighbors,
		Real smoothingLength,
		Kernel weight,
		Real scale)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;
		if (!attribute[pId].isDynamic()) return;

		Coord pos_i = position[pId];
		Real alpha_i = 0.0f;

		List<int>& list_i = neighbors[pId];
		int nbSize = list_i.size();
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = list_i[ne];
			Real r = (pos_i - position[j]).norm();;

			if (r > EPSILON)
			{
				Real a_ij = weight(r, smoothingLength, scale);
				alpha_i += a_ij;
			}
		}

		alpha[pId] = alpha_i;
	}

	template <typename Real>
	__global__ void VAP_CorrectAlpha
	(
		DArray<Real> alpha,
		Real maxAlpha)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= alpha.size()) return;

		Real alpha_i = alpha[pId];
		if (alpha_i < maxAlpha)
		{
			alpha_i = maxAlpha;
		}
		alpha[pId] = alpha_i;
	}

	template <typename Real, typename Coord, typename Kernel>
	__global__ void VAP_ComputeDiagonalElement
	(
		DArray<Real> AiiFluid,
		DArray<Real> AiiTotal,
		DArray<Real> alpha,
		DArray<Coord> position,
		DArray<Attribute> attribute,
		DArrayList<int> neighbors,
		Real smoothingLength,
		Kernel kernWRR,
		Real scale
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;
		if (!attribute[pId].isDynamic()) return;

		Real invAlpha = 1.0f / alpha[pId];


		Real diaA_total = 0.0f;
		Real diaA_fluid = 0.0f;
		Coord pos_i = position[pId];

		List<int>& list_i = neighbors[pId];
		int nbSize = list_i.size();
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = list_i[ne];
			Real r = (pos_i - position[j]).norm();

			Attribute att_j = attribute[j];
			if (r > EPSILON)
			{
				Real wrr_ij = invAlpha * kernWRR(r, smoothingLength, scale);
				if (att_j.isDynamic())
				{
					diaA_total += wrr_ij;
					diaA_fluid += wrr_ij;
					atomicAdd(&AiiFluid[j], wrr_ij);
					atomicAdd(&AiiTotal[j], wrr_ij);
				}
				else
				{
					diaA_total += 2.0f * wrr_ij;
				}
			}
		}

		atomicAdd(&AiiFluid[pId], diaA_fluid);
		atomicAdd(&AiiTotal[pId], diaA_total);
	}

	template <typename Real, typename Coord, typename Kernel>
	__global__ void VAP_ComputeDiagonalElement
	(
		DArray<Real> diaA,
		DArray<Real> alpha,
		DArray<Coord> position,
		DArray<Attribute> attribute,
		DArrayList<int> neighbors,
		Real smoothingLength,
		Kernel kernWRR,
		Real scale
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;
		if (!attribute[pId].isDynamic()) return;

		Coord pos_i = position[pId];
		Real invAlpha_i = 1.0f / alpha[pId];
		Real A_i = 0.0f;

		List<int>& list_i = neighbors[pId];
		int nbSize = list_i.size();
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = list_i[ne];
			Real r = (pos_i - position[j]).norm();

			if (r > EPSILON && attribute[j].isDynamic())
			{
				Real wrr_ij = invAlpha_i * kernWRR(r, smoothingLength, scale);
				A_i += wrr_ij;
				atomicAdd(&diaA[j], wrr_ij);
			}
		}

		atomicAdd(&diaA[pId], A_i);
	}

	template <typename Real, typename Coord>
	__global__ void VAP_DetectSurface
	(
		DArray<Real> Aii,
		DArray<bool> bSurface,
		DArray<Real> AiiFluid,
		DArray<Real> AiiTotal,
		DArray<Coord> position,
		DArray<Attribute> attribute,
		DArrayList<int> neighbors,
		Real smoothingLength,
		Real maxA
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;
		if (!attribute[pId].isDynamic()) return;

		Real total_weight = 0.0f;
		Coord div_i = Coord(0);

		SmoothKernel<Real> kernSmooth;

		Coord pos_i = position[pId];
		bool bNearWall = false;

		List<int>& list_i = neighbors[pId];
		int nbSize = list_i.size();
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = list_i[ne];
			Real r = (pos_i - position[j]).norm();

			if (r > EPSILON && attribute[j].isDynamic())
			{
				float weight = -kernSmooth.Gradient(r, smoothingLength);
				total_weight += weight;
				div_i += (position[j] - pos_i) * (weight / r);
			}

			if (!attribute[j].isDynamic())
			{
				bNearWall = true;
			}
		}

		total_weight = total_weight < EPSILON ? 1.0f : total_weight;
		Real absDiv = div_i.norm() / total_weight;

		bool bSurface_i = false;
		Real diagF_i = AiiFluid[pId];
		Real diagT_i = AiiTotal[pId];
		Real aii = diagF_i;
		Real diagS_i = diagT_i - diagF_i;

		//A hack, further improvements should be done to impose the exact solid boundary condition
		Real threshold = 1.5f;
		if (bNearWall && diagT_i < threshold * maxA)
		{
			bSurface_i = true;
			aii = threshold * maxA - (diagT_i - diagF_i);
		}

		if (!bNearWall && diagF_i < maxA)
		{
			bSurface_i = true;
			aii = maxA;
		}
		bSurface[pId] = bSurface_i;
		Aii[pId] = aii;
	}

	template <typename Real, typename Coord, typename Kernel>
	__global__ void VAP_ComputeDivergence
	(
		DArray<Real> divergence,
		DArray<Real> alpha,
		DArray<Real> density,
		DArray<Coord> position,
		DArray<Coord> velocity,
		DArray<bool> bSurface,
		DArray<Coord> normals,
		DArray<Attribute> attribute,
		DArrayList<int> neighbors,
		Real separation,
		Real tangential,
		Real restDensity,
		Real smoothingLength,
		Real dt,
		Kernel gradient,
		Real scale
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;
		if (!attribute[pId].isDynamic()) return;

		Coord pos_i = position[pId];
		Coord vel_i = velocity[pId];

		Real invAlpha_i = 1.0f / alpha[pId];

		List<int>& list_i = neighbors[pId];
		int nbSize = list_i.size();
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = list_i[ne];
			Real r = (pos_i - position[j]).norm();

			if (r > EPSILON && attribute[j].isFluid())
			{
				Real wr_ij = gradient(r, smoothingLength, scale);
				Coord g = -invAlpha_i * (pos_i - position[j]) * wr_ij * (1.0f / r);

				if (attribute[j].isDynamic())
				{
					Real div_ij = 0.5f * (vel_i - velocity[j]).dot(g) * restDensity / dt;	//dv_ij = 1 / alpha_i * (v_i-v_j).*(x_i-x_j) / r * (w / r);
					atomicAdd(&divergence[pId], div_ij);
					atomicAdd(&divergence[j], div_ij);
				}
				else
				{
					//float div_ij = dot(2.0f*vel_i, g)*const_vc_state.restDensity / dt;
					Coord normal_j = normals[j];

					Coord dVel = vel_i - velocity[j];
					Real magNVel = dVel.dot(normal_j);
					Coord nVel = magNVel * normal_j;
					Coord tVel = dVel - nVel;

					//float div_ij = dot(2.0f*dot(vel_i - velArr[j], normal_i)*normal_i, g)*const_vc_state.restDensity / dt;
					if (magNVel < -EPSILON)
					{
						Real div_ij = g.dot(2.0f * (nVel + tangential * tVel)) * restDensity / dt;

						atomicAdd(&divergence[pId], div_ij);
					}
					else
					{
						Real div_ij = g.dot(2.0f * (separation * nVel + tangential * tVel)) * restDensity / dt;
						atomicAdd(&divergence[pId], div_ij);
					}

				}
			}
		}

	}

	template <typename Real, typename Coord>
	__global__ void VAP_CompensateSource
	(
		DArray<Real> divergence,
		DArray<Real> density,
		DArray<Attribute> attribute,
		DArray<Coord> position,
		Real restDensity,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= density.size()) return;
		if (!attribute[pId].isDynamic()) return;

		//divergence[pId] *= 1000.0f;

		Coord pos_i = position[pId];
		if (density[pId] > restDensity)
		{
			Real ratio = (density[pId] - restDensity) / restDensity;
			atomicAdd(&divergence[pId], 5.0 * restDensity * ratio / (dt * dt));
		}
	}

	// compute Ax;
	template <typename Real, typename Coord, typename Kernel>
	__global__ void VAP_ComputeAx
	(
		DArray<Real> residual,
		DArray<Real> pressure,
		DArray<Real> aiiSymArr,
		DArray<Real> alpha,
		DArray<Coord> position,
		DArray<Attribute> attribute,
		DArrayList<int> neighbors,
		Real smoothingLength,
		Kernel kernWRR,
		Real scale
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;
		if (!attribute[pId].isDynamic()) return;

		Coord pos_i = position[pId];
		Real invAlpha_i = 1.0f / alpha[pId];

		atomicAdd(&residual[pId], aiiSymArr[pId] * pressure[pId]);
		Real con1 = 1.0f;// PARAMS.mass / PARAMS.restDensity / PARAMS.restDensity;

		List<int>& list_i = neighbors[pId];
		int nbSize = list_i.size();
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = list_i[ne];
			Real r = (pos_i - position[j]).norm();

			if (r > EPSILON && attribute[j].isDynamic())
			{
				Real wrr_ij = kernWRR(r, smoothingLength, scale);
				Real a_ij = -invAlpha_i * wrr_ij;
				//				residual += con1*a_ij*preArr[j];
				atomicAdd(&residual[pId], con1 * a_ij * pressure[j]);
				atomicAdd(&residual[j], con1 * a_ij * pressure[pId]);
			}
		}
	}

	template <typename Real, typename Coord, typename Kernel>
	__global__ void VAP_UpdateVelocity1rd
	(
		DArray<Real> preArr,
		DArray<Real> aiiArr,
		DArray<bool> bSurface,
		DArray<Coord> posArr,
		DArray<Coord> velArr,
		DArray<Attribute> attArr,
		DArrayList<int> neighbors,
		Real restDensity,
		Real smoothingLength,
		Real dt,
		Kernel gradient,
		Real scale
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.size()) return;

		CorrectedMPSKernel<Real> Mpskernel;

		if (attArr[pId].isDynamic())
		{
			Coord pos_i = posArr[pId];
			Coord b = Coord(0.0f);
			//glm::mat3 A_i = glm::mat3(0.0f);
			float p_i = preArr[pId];
			bool bNearWall = false;

			float total_weight = 0.0f;
			List<int>& list_i = neighbors[pId];
			int nbSize = list_i.size();

			//A_i = glm::mat3();

			float invAii = 1.0f / aiiArr[pId];
			Coord vel_i = velArr[pId];
			Coord dv_i = Coord(0.0f);
			float scale = 0.1f * dt / restDensity;
			for (int ne = 0; ne < nbSize; ne++)
			{
				int j = list_i[ne];
				Real r = (pos_i - posArr[j]).norm();

				Attribute att_j = attArr[j];
				if (r > EPSILON)
				{
					Real weight = -invAii * gradient(r, smoothingLength, scale);
					Coord dnij = -scale * (pos_i - posArr[j]) * weight * (1.0f / r);
					Coord corrected = dnij;// Vec2Float(A_i*Float2Vec(dnij));

					Real clamp_r = r;
					Coord dvij = (preArr[j] - preArr[pId]) * corrected;
					Coord dvjj = (preArr[j] +/* const_vc_state.pAir*/0) * corrected;

					//Calculate asymmetric pressure force
					if (att_j.isDynamic())
					{
						if (bSurface[pId] && !bNearWall)
						{
							dv_i += dvjj;
						}
						else
						{
							dv_i += dvij;
						}
					}
					else
					{
						Real weight = 1.0f * invAii * Mpskernel.WeightRR(r, smoothingLength);
						Coord nij = (pos_i - posArr[j]);
						dv_i += weight * (velArr[j] - vel_i).dot(nij) * nij;
					}

					//Stabilize particles under compression state.
					if (preArr[j] + preArr[pId] > 0.0f)
					{
						Real clamp_r = r;
						Coord dvij = (preArr[pId]) * dnij;

						if (att_j.isDynamic())
						{
							atomicAdd(&velArr[j].x, -dvij.x);
							atomicAdd(&velArr[j].y, -dvij.y);
							atomicAdd(&velArr[j].z, -dvij.z);
						}
						atomicAdd(&velArr[pId].x, dvij.x);
						atomicAdd(&velArr[pId].y, dvij.y);
						atomicAdd(&velArr[pId].z, dvij.z);
					}
				}
			}

			velArr[pId] += dv_i;
		}
	}


	template<typename Coord>
	__global__ void VC_InitializeNormal(
		DArray<Coord> normal)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= normal.size()) return;

		normal[pId] = Coord(0);
	}

	__global__ void VC_InitializeAttribute(
		DArray<Attribute> attr)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= attr.size()) return;

		attr[pId].setDynamic();
	}


	template<typename TDataType>
	void VariationalApproximateProjection<TDataType>::varChanged()
	{
		int num = this->inPosition()->size();
		this->resizeArray(num);
	}

	template<typename TDataType>
	void VariationalApproximateProjection<TDataType>::resizeArray(int num)
	{
		/*
		*@Note	If particle size is not constant (Particle emitter is used), and the particle attribute/normal is not prepared,
		*		the solid particles will be transformed to fluid particles.
		*/
		if ((this->inAttribute()->isEmpty()) || (this->inNormal()->isEmpty())
			|| (this->inAttribute()->size() != num) || (this->inNormal()->size() != num))
		{
			this->inAttribute()->resize(num);

			this->inNormal()->resize(num);

			cuExecute(num,
				VC_InitializeNormal,
				this->inNormal()->getData());

			cuExecute(num,
				VC_InitializeAttribute,
				this->inAttribute()->getData());
		}

		mAlpha.resize(num);
		mAii.resize(num);
		mAiiFluid.resize(num);
		mAiiTotal.resize(num);
		mPressure.resize(num);
		mDivergence.resize(num);
		mIsSurface.resize(num);
		mPressure.resize(num);

		m_y.resize(num);
		m_r.resize(num);
		m_p.resize(num);

		m_reduce = Reduction<float>::Create(num);
		m_arithmetic = Arithmetic<float>::Create(num);

		mAlpha.reset();
		cuZerothOrder(num, this->varKernelType()->getDataPtr()->currentKey(), this->mScalingFactor,
			VAP_ComputeAlpha,
			mAlpha,
			this->inPosition()->getData(),
			this->inAttribute()->getData(),
			this->inNeighborIds()->getData(),
			this->inSmoothingLength()->getData()
		);

		mAlphaMax = m_reduce->maximum(mAlpha.begin(), mAlpha.size());

		cuExecute(num,
			VAP_CorrectAlpha,
			mAlpha,
			mAlphaMax);

		mAiiFluid.reset();
		cuSecondOrder(num, this->varKernelType()->getDataPtr()->currentKey(), this->mScalingFactor,
			VAP_ComputeDiagonalElement,
			mAiiFluid,
			mAlpha,
			this->inPosition()->getData(),
			this->inAttribute()->getData(),
			this->inNeighborIds()->getData(),
			this->inSmoothingLength()->getData());

		mAMax = m_reduce->maximum(mAiiFluid.begin(), mAiiFluid.size());

		std::cout << "Max alpha: " << mAlphaMax << std::endl;
		std::cout << "Max A: " << mAMax << std::endl;
	}

	template<typename TDataType>
	void VariationalApproximateProjection<TDataType>::compute()
	{
		int num = this->inPosition()->size();

		if (mAlpha.size() != num)
		{
			this->resizeArray(num);
		}
		Real dt = this->inTimeStep()->getValue();
	
		mAlpha.reset();

		cuZerothOrder(num, this->varKernelType()->getDataPtr()->currentKey(), this->mScalingFactor,
			VAP_ComputeAlpha,
			mAlpha,
			this->inPosition()->getData(),
			this->inAttribute()->getData(),
			this->inNeighborIds()->getData(),
			this->inSmoothingLength()->getValue());

		cuExecute(num,
			VAP_CorrectAlpha,
			mAlpha,
			mAlphaMax);

		//compute the diagonal elements of the coefficient matrix
		mAiiFluid.reset();
		mAiiTotal.reset();

		cuSecondOrder(num, this->varKernelType()->getDataPtr()->currentKey(), this->mScalingFactor,
			VAP_ComputeDiagonalElement,
			mAiiFluid,
			mAiiTotal,
			mAlpha,
			this->inPosition()->getData(),
			this->inAttribute()->getData(),
			this->inNeighborIds()->getData(),
			this->inSmoothingLength()->getData());

		mIsSurface.reset();
		mAii.reset();

		cuExecute(num,
			VAP_DetectSurface,
			mAii,
			mIsSurface,
			mAiiFluid,
			mAiiTotal,
			this->inPosition()->getData(),
			this->inAttribute()->getData(),
			this->inNeighborIds()->getData(),
			this->inSmoothingLength()->getData(),
			mAMax);

		int itor = 0;

		mPressure.reset();

		//compute the source term
		mDensityCalculator->compute();
		mDivergence.reset();
		cuFirstOrder(num, this->varKernelType()->getDataPtr()->currentKey(), this->mScalingFactor,
			VAP_ComputeDivergence,
			mDivergence,
			mAlpha,
			mDensityCalculator->outDensity()->getData(),
			this->inPosition()->getData(),
			this->inVelocity()->getData(),
			mIsSurface,
			this->inNormal()->getData(),
			this->inAttribute()->getData(),
			this->inNeighborIds()->getData(),
			mSeparation,
			mTangential,
			this->varRestDensity()->getData(),
			this->inSmoothingLength()->getData(),
			dt);

		cuExecute(num,
			VAP_CompensateSource,
			mDivergence,
			mDensityCalculator->outDensity()->getData(),
			this->inAttribute()->getData(),
			this->inPosition()->getData(),
			this->varRestDensity()->getData(),
			dt);

		//solve the linear system of equations with a conjugate gradient method.
		m_y.reset();

		cuSecondOrder(num, this->varKernelType()->getDataPtr()->currentKey(), this->mScalingFactor,
			VAP_ComputeAx,
			m_y,
			mPressure,
			mAii,
			mAlpha,
			this->inPosition()->getData(),
			this->inAttribute()->getData(),
			this->inNeighborIds()->getData(),
			this->inSmoothingLength()->getData());

		m_r.reset();
		Function2Pt::subtract(m_r, mDivergence, m_y);
		m_p.assign(m_r);
		Real rr = m_arithmetic->Dot(m_r, m_r);
		Real err = sqrt(rr / m_r.size());

		while (itor < 1000 && err > 1.0f)
		{
			m_y.reset();
			//VC_ComputeAx << <pDims, BLOCK_SIZE >> > (*yArr, *pArr, *aiiArr, *alphaArr, *posArr, *attArr, *neighborArr);
			cuSecondOrder(num, this->varKernelType()->getDataPtr()->currentKey(), this->mScalingFactor,
				VAP_ComputeAx,
				m_y,
				m_p,
				mAii,
				mAlpha,
				this->inPosition()->getData(),
				this->inAttribute()->getData(),
				this->inNeighborIds()->getData(),
				this->inSmoothingLength()->getValue());

			float alpha = rr / m_arithmetic->Dot(m_p, m_y);
			Function2Pt::saxpy(mPressure, m_p, mPressure, alpha);
			Function2Pt::saxpy(m_r, m_y, m_r, -alpha);

			Real rr_old = rr;

			rr = m_arithmetic->Dot(m_r, m_r);

			Real beta = rr / rr_old;
			Function2Pt::saxpy(m_p, m_p, m_r, beta);

			err = sqrt(rr / m_r.size());

			itor++;
		}

		//update the each particle's velocity
// 		VC_UpdateVelocityBoundaryCorrected << <pDims, BLOCK_SIZE >> > (
// 			m_pressure,
// 			m_alpha,
// 			m_bSurface, 
// 			this->inPosition()->getData(),
// 			this->inVelocity()->getData(),
// 			this->inNormal()->getData(),
// 			this->inAttribute()->getData(),
// 			this->inNeighborIds()->getData(),
// 			m_restDensity,
// 			m_airPressure,
// 			m_tangential,
// 			m_separation,
// 			this->inSmoothingLength()->getData(),
// 			dt);


		cuFirstOrder(num, this->varKernelType()->getDataPtr()->currentKey(), this->mScalingFactor,
			VAP_UpdateVelocity1rd,
			mPressure,
			mAlpha,
			mIsSurface,
			this->inPosition()->getData(),
			this->inVelocity()->getData(),
			this->inAttribute()->getData(),
			this->inNeighborIds()->getData(),
			this->varRestDensity()->getValue(),
			this->inSmoothingLength()->getValue(),
			dt);

	}

	DEFINE_CLASS(VariationalApproximateProjection);
}