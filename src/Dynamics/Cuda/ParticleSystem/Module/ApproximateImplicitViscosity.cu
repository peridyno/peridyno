#include "ApproximateImplicitViscosity.h"
//#include <string>
#include "Algorithm/Function2Pt.h"


namespace dyno
{

	template<typename TDataType>
	ApproximateImplicitViscosity<TDataType>::ApproximateImplicitViscosity()
		:ConstraintModule()
	{
		this->inAttribute()->tagOptional(true);
	}


	//F-norms
	template<typename Real>
	__device__ Real VB_FNorm(const Real a, const Real b, const Real c)
	{
		Real p = a + b + c;
		return sqrt(p * p / 2);
	}

	//CrossModel Viscosity Coefficient
	template<typename Real>
	__device__ Real VB_Viscosity(const Real Viscosity_h, const Real Viscosity_l, const Real StrainRate, const Real CrossModel_K, const Real CrossModel_n)
	{
		Real p = CrossModel_K * StrainRate;
		p = pow(p, CrossModel_n);
		return Viscosity_h + (Viscosity_l - Viscosity_h) / (1 + p);
	}

	__device__ inline float kernWeight(const float r, const float h)
	{
		const float q = r / h;
		if (q > 1.0f) return 0.0f;
		else {	
			return (1.0 - pow(q, 4.0f));
		}
	}

	__device__ inline float kernWR(const float r, const float h)
	{
		float w = kernWeight(r, h);
		const float q = r / h;
		if (q < 0.4f)
		{
			return w / (0.4f*h);
		}
		return w / r;
	}

	__device__ inline float kernWRR(const float r, const float h)
	{
		float w = kernWeight(r, h);
		const float q = r / h;
		if (q < 0.4f)
		{
			return w / (0.16f*h*h);
		}
		return w / r / r;
	}

	template <typename Real, typename Coord>
	__global__ void VC_ComputeAlpha
	(
		DArray<Real> alpha,
		DArray<Coord> position,
		DArray<Attribute> attribute,
		DArrayList<int> neighbors,
		Real smoothingLength
	)
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
				Real a_ij = kernWeight(r, smoothingLength);
				alpha_i += a_ij;
			}
		}
		alpha[pId] = alpha_i;
	}

	template <typename Real>
	__global__ void VC_CorrectAlpha
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

	//Jacobi Mehthod 
	template <typename Real, typename Coord, typename Matrix>
	__global__ void VC_VisComput
	(
		DArray<Coord> velNew,
		DArray<Coord> velBuf,
		DArray<Coord> velOld,
		DArray<Coord> velDp,
		DArray<Coord> position,
		DArray<Real> alpha,
		DArray<Attribute> attribute,
		DArrayList<int> neighbor,
		Real rest_density,
		Real h,
		Real dt,
		DArray<Real> vis
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;
		if (!attribute[pId].isDynamic()) return;

		Real tempValue = 0;
		Coord Avj(0);
		Matrix Mii(0);
		Real invAlpha_i = 1.0f / alpha[pId];

		List<int>& list_i = neighbor[pId];
		int nbSize = list_i.size();
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = list_i[ne];
			Real r = (position[pId] - position[j]).norm();
			if (r > EPSILON)
			{
				Real wrr_ij = kernWRR(r, h);
				Real ai = 1.0f / alpha[pId];
				Real aj = 1.0f / alpha[j];
				Coord nij = (position[j] - position[pId]) / r;
				tempValue = 0.25 * dt * (vis[pId] + vis[j]) * (ai + aj) * wrr_ij / rest_density;
				Avj += tempValue * velBuf[j].dot(nij) * nij;
				Matrix Mij(0);
				Mij(0, 0) += nij[0] * nij[0];	Mij(0, 1) += nij[0] * nij[1];	Mij(0, 2) += nij[0] * nij[2];
				Mij(1, 0) += nij[1] * nij[0];	Mij(1, 1) += nij[1] * nij[1];	Mij(1, 2) += nij[1] * nij[2];
				Mij(2, 0) += nij[2] * nij[0];	Mij(2, 1) += nij[2] * nij[1];	Mij(2, 2) += nij[2] * nij[2];
				Mii += Mij*tempValue;
			}
		}
		Mii += Matrix::identityMatrix();
		velNew[pId] = Mii.inverse()*(velOld[pId] + velDp[pId] + Avj);
	}


	//Conjugate Gradient Method.
	template <typename Real, typename Coord>
	__global__ void VC_Vis_AxComput
	(
		DArray<Real> v_y,
		DArray<Coord> velBuf,
		DArray<Coord> position,
		DArray<Real> alpha,
		DArray<Attribute> attribute,
		DArrayList<int> neighbor,
		Real rest_density,
		Real h,
		Real dt,
		DArray<Real> vis
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;
		if (!attribute[pId].isDynamic()) return;

		Real tempValue = 0;
		Coord Avi(0);
		Real invAlpha_i = 1.0f / alpha[pId];

		List<int>& list_i = neighbor[pId];
		int nbSize = list_i.size();
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = list_i[ne];
			Real r = (position[pId] - position[j]).norm();
			if (r > EPSILON)
			{
				Real wrr_ij = kernWRR(r, h);
				Real ai = 1.0f / alpha[pId];
				Real aj = 1.0f / alpha[j];
				Coord nij = (position[j] - position[pId]) / r;
				Coord vij = velBuf[pId] - velBuf[j];
				tempValue = 2.5 * dt * (vis[pId] + vis[j]) * (ai + aj) * wrr_ij / rest_density; 
				Avi += tempValue * vij.dot(nij) * nij;
			}
		}
		Avi += velBuf[pId];
		v_y[3 * pId] = Avi[0];
		v_y[3 * pId + 1] = Avi[1];
		v_y[3 * pId + 2] = Avi[2];
	}


	template <typename Real, typename Coord>
	__global__ void VC_Vis_r_Comput
	(
		DArray<Real> v_r,
		DArray<Real> v_y,
		DArray<Coord> vel_old,
		DArray<Coord> position,
		DArray<Attribute> attribute
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;
		if (!attribute[pId].isDynamic()) return;
		Coord temp_vel = vel_old[pId];
		v_r[3 * pId] = temp_vel[0] - v_y[3 * pId];
		v_r[3 * pId + 1] = temp_vel[1] - v_y[3 * pId + 1];
		v_r[3 * pId + 2] = temp_vel[2] - v_y[3 * pId + 2];
	}

	template <typename Real, typename Coord>
	__global__ void VC_Vis_pToVector
	(
		DArray<Real> v_p,
		DArray<Coord> pv,
		DArray<Attribute> attribute
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= attribute.size()) return;
		if (!attribute[pId].isDynamic()) return;
		pv[pId][0] = v_p[3 * pId];
		pv[pId][1] = v_p[3 * pId + 1];
		pv[pId][2] = v_p[3 * pId + 2];
	}


	template <typename Real, typename Coord>
	__global__ void VC_Vis_CoordToReal
	(
		DArray<Real> veloReal,
		DArray<Coord> vel,
		DArray<Attribute> attribute
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= attribute.size()) return;
		if (!attribute[pId].isDynamic()) return;
		veloReal[3 * pId] = vel[pId][0];
		veloReal[3 * pId + 1] = vel[pId][1];
		veloReal[3 * pId + 2] = vel[pId][2];
	}

	template <typename Real, typename Coord>
	__global__ void VC_Vis_RealToVeloctiy
	(
		DArray<Coord> vel,
		DArray<Real> veloReal,
		DArray<Attribute> attribute
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= vel.size()) return;
		if (!attribute[pId].isDynamic()) return;
		vel[pId][0] = veloReal[3 * pId];
		vel[pId][1] = veloReal[3 * pId + 1];
		vel[pId][2] = veloReal[3 * pId + 2];
	}

	template <typename Real, typename Coord>
	__global__ void VC_CrossVis
	(
		DArray<Real> crossVis,
		DArray<Coord> velBuf,
		DArray<Coord> position,
		DArray<Real> alpha,
		DArray<Attribute> attribute,
		DArrayList<int> neighbor,
		Real smoothingLength,
		Real visTop,
		Real visFloor,
		Real K,
		Real N
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;
		if (!attribute[pId].isDynamic()) return;				

		Coord pos_i = position[pId];
		Coord vel_i = velBuf[pId];
		Real invAlpha_i = 1.0f / alpha[pId];

		Coord dv(0);
		Coord vij(0);

		List<int>& list_i = neighbor[pId];
		int nbSize = list_i.size();
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = list_i[ne];
			Real r = (position[pId] - position[j]).norm();
			if(r > EPSILON)
			{
				Real wr_ij = kernWR(r, smoothingLength);
				Coord g = -invAlpha_i*(pos_i - position[j])*wr_ij*(1.0f / r);
				vij = 0.5*(velBuf[pId] - velBuf[j]);
				dv[0] += vij[0] * g[0];
				dv[1] += vij[1] * g[1];
				dv[2] += vij[2] * g[2];
			}

		}
		Real Norm = VB_FNorm(dv[0], dv[1], dv[2]);
		crossVis[pId] = VB_Viscosity(visTop, visFloor, Norm, K, N);
		if (pId == 100) printf("viscosity : %f\r\n", crossVis[pId]);
	}

	template <typename Real, typename Coord>
	__global__ void VC_updateScar
	(
		DArray<Real> scar,
		DArray<Coord> velocity
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= velocity.size()) return;
		
		scar[pId] = velocity[pId].norm();
		
	}
	
	__global__ void VC_AttributeInit
	(
		DArray<Attribute> atts
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= atts.size()) return;

		atts[pId].setFluid();
		atts[pId].setDynamic();
	}

	template <typename Real>
	__global__ void VC_ViscosityValueUpdate
	(
		DArray<Real> viscosities,
		Real viscosityValue
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= viscosities.size()) return;
		viscosities[pId] = viscosityValue;
	}

	template<typename TDataType>
	ApproximateImplicitViscosity<TDataType>::~ApproximateImplicitViscosity()
	{
		m_alpha.clear();
		//if (m_reduce)
		//{
		//	delete m_reduce;
		//}
	};


	template<typename TDataType>
	bool ApproximateImplicitViscosity<TDataType>::SetCross()
	{
		CrossVisCeil = this->varViscosity()->getValue();
		CrossVisFloor = this->varLowerBoundViscosity()->getValue();
		CrossVisFloor = CrossVisFloor < CrossVisCeil ? CrossVisFloor : CrossVisCeil;

		Cross_K = this->varCrossK()->getValue();
		Cross_N = this->varCrossN()->getValue();
		std::cout << "*Non-Newtonian Fluid, Viscosity:" << CrossVisFloor
			<< " to " << CrossVisCeil << ", Cross_K:"
			<< Cross_K << ", Cross_N:" << Cross_N
			<< std::endl;
		return true;
	};

	template<typename TDataType>
	void ApproximateImplicitViscosity<TDataType>::constrain()
	{
		std::cout << "*Approximate Vicosity Solver::Dynamic Viscosity: " << this->varViscosity()->getValue()  << std::endl;

		int num = this->inPosition()->size();

		if ((num != m_alpha.size())||
			(num != velOld.size()) ||
			(num != m_viscosity.size()))
		{
			m_alpha.resize(num);
			v_y.resize(3 * num);
			v_r.resize(3 * num);
			v_p.resize(3 * num);
			v_pv.resize(num);
			m_VelocityReal.resize(3 * num);
			velOld.resize(num);
			velBuf.resize(num);
			m_viscosity.resize(num);
			m_reduce = Reduction<float>::Create(num);
			m_arithmetic_v = Arithmetic<float>::Create(3 * num);
		}

		Real dt = this->inTimeStep()->getValue();

		if (this->inAttribute()->isEmpty() || this->inAttribute()->size()!= this->inPosition()->size())
		{
			this->inAttribute()->allocate();
			this->inAttribute()->resize(this->inPosition()->size());
			cuExecute(num, VC_AttributeInit, this->inAttribute()->getData());
		}

		auto& m_position = this->inPosition()->getData();
		auto& m_velocity = this->inVelocity()->getData();
		auto& m_neighborhood = this->inNeighborIds()->getData();
		auto& m_attribute = this->inAttribute()->getData();
		auto m_smoothingLength = this->varSmoothingLength()->getValue();
		auto m_restDensity = this->varRestDensity()->getValue();

		cuExecute(num, VC_ViscosityValueUpdate,
			m_viscosity,
			this->varViscosity()->getValue()
			);

		m_alpha.reset();
		cuExecute(num, VC_ComputeAlpha,
			m_alpha,
			m_position,
			m_attribute,
			m_neighborhood,
			m_smoothingLength);

		//VC_CorrectAlpha << <pDims, BLOCK_SIZE >> > (
		//	m_alpha,
		//	m_maxAlpha);

		//IF Cross Model is active, viscous coefficients should be computed.


		if (this->varFluidType()->getValue() == FluidType::NonNewtonianFluid)
		{
			SetCross();
			cuExecute(num, VC_CrossVis,
				m_viscosity,
				m_velocity,
				m_position,
				m_alpha,
				m_attribute,
				m_neighborhood,
				m_smoothingLength,
				CrossVisCeil,
				CrossVisFloor,
				Cross_K,
				Cross_N
				);
		}

		velOld.assign(m_velocity);
		v_y.reset();

		cuExecute(num, VC_Vis_AxComput,
				v_y,
				m_velocity,
				m_position,
				m_alpha,
				m_attribute,
				m_neighborhood,
				m_restDensity,
				m_smoothingLength,
				dt,
				m_viscosity
				);

		v_r.reset();
		cuExecute(num, VC_Vis_r_Comput,
				v_r,
				v_y,
				velOld,
				m_position,
				m_attribute
				);

		v_p.assign(v_r);

		Real Vrr = m_arithmetic_v->Dot(v_r, v_r);
		Real Verr = sqrt(Vrr / v_r.size());
		int VisItor = 0;
		Real initErr = Verr;

		std::cout <<"*Approximate Vicosity Solver::Residual:"  << Vrr <<std::endl;
		while (VisItor < 1000 && Verr / initErr > 0.01f && Vrr > 1000.0f * EPSILON)
		{
				VisItor++;
		//		//The type of "v_p" should convert to DArray<Coord>
				cuExecute(num, VC_Vis_pToVector,
					v_p,
					v_pv,
					m_attribute
				);

				v_y.reset();
				cuExecute(num, VC_Vis_AxComput,
					v_y,
					v_pv,
					m_position,
					m_alpha,
					m_attribute,
					m_neighborhood,
					m_restDensity,
					m_smoothingLength,
					dt,
					m_viscosity
					);

				float alpha = Vrr / m_arithmetic_v->Dot(v_p, v_y);

				cuExecute(num, VC_Vis_CoordToReal,
					m_VelocityReal,
					m_velocity,
					m_attribute
					);

				Function2Pt::saxpy(m_VelocityReal, v_p, m_VelocityReal, alpha);

				cuExecute(num, VC_Vis_RealToVeloctiy,
					m_velocity,
					m_VelocityReal,
					m_attribute
					);

				Function2Pt::saxpy(v_r, v_y, v_r, -alpha);
				Real Vrr_old = Vrr;

				Vrr = m_arithmetic_v->Dot(v_r, v_r);
				Real beta = Vrr / Vrr_old;
				Function2Pt::saxpy(v_p, v_p, v_r, beta);

				Verr = sqrt(Vrr / v_r.size());
		}
		std::cout << "*Approximate Vicosity Solver::Iteration #" << VisItor << ", Relative Resisual:" << Verr / initErr << std::endl;
	};

	DEFINE_CLASS(ApproximateImplicitViscosity);
}