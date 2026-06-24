#include "ApproximateImplicitViscosity.h"
//#include <string>
#include "Algorithm/Function2Pt.h"


namespace dyno
{

	template<typename TDataType>
	ApproximateImplicitViscosity<TDataType>::ApproximateImplicitViscosity()
		:ConstraintModule(),
		m_reduce(),
		m_arithmetic_v()
	{
		this->inAttribute()->tagOptional(true);
	}


	template<typename Real>
	__device__ inline Real vb_trace_norm(const Real a, const Real b, const Real c)
	{
		Real p = a + b + c;
		return sqrt(p * p / 2);
	}

	template<typename Real>
	__device__ inline Real vb_cross_viscosity(const Real Viscosity_h, const Real Viscosity_l, const Real StrainRate, const Real CrossModel_K, const Real CrossModel_n)
	{
		Real p = CrossModel_K * StrainRate;
		p = pow(p, CrossModel_n);
		return Viscosity_h + (Viscosity_l - Viscosity_h) / (1 + p);
	}

	template<typename Real>
	__device__ inline Real kernWeight(const float r, const float h)
	{
		if (r / h > 1.0f) return 0.0f;
		else {	
			return (1.0 - pow(r / h, 4.0f));
		}
	}

	template<typename Real>
	__device__ inline Real kernWR(const float r, const float h)
	{
		if (r / h < 0.4f)
		{
			return kernWeight<Real>(r, h) / (0.4f*h);
		}
		return kernWeight<Real>(r, h) / r;
	}

	__device__ inline float kernWRR(const float r, const float h)
	{
		Real w = kernWeight<Real>(r, h);
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
		for (int ne = 0; ne < list_i.size(); ne++)
		{
			int j = list_i[ne];
			Real r = (pos_i - position[j]).norm();;
			if (r > EPSILON)
			{
				Real a_ij = kernWeight<Real>(r, smoothingLength);
				alpha_i += a_ij;
			}
		}
		alpha[pId] = alpha_i;
	}

	//Conjugate Gradient Method.
	template <typename Real, typename Coord>
	__global__ void VC_Vis_AxComput
	(
		DArray<Real> vy,
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
				Avi += 2.5 * dt * (vis[pId] + vis[j]) * (ai + aj) * wrr_ij  * (velBuf[pId] - velBuf[j]).dot(nij) * nij / rest_density;;
			}
		}
		Avi += velBuf[pId];
		vy[3 * pId] = Avi[0];
		vy[3 * pId + 1] = Avi[1];
		vy[3 * pId + 2] = Avi[2];
	}


	template <typename Real, typename Coord>
	__global__ void VC_Vis_r_Comput
	(
		DArray<Real> vr,
		DArray<Real> vy,
		DArray<Coord> vel_old,
		DArray<Coord> position,
		DArray<Attribute> attribute
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;
		if (!attribute[pId].isDynamic()) return;

		vr[3 * pId] = vel_old[pId][0] - vy[3 * pId];
		vr[3 * pId + 1] = vel_old[pId][1] - vy[3 * pId + 1];
		vr[3 * pId + 2] = vel_old[pId][2] - vy[3 * pId + 2];
	}

	template <typename Real, typename Coord>
	__global__ void VC_Vis_pToVector
	(
		DArray<Real> vp,
		DArray<Coord> pv,
		DArray<Attribute> attribute
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= attribute.size()) return;
		if (!attribute[pId].isDynamic()) return;
		pv[pId][0] = vp[3 * pId];
		pv[pId][1] = vp[3 * pId + 1];
		pv[pId][2] = vp[3 * pId + 2];
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

		Coord dv(0), vij(0);

		List<int>& list_i = neighbor[pId];
		for (int ne = 0; ne < list_i.size(); ne++)
		{
			int j = list_i[ne];
			Real r = (position[pId] - position[j]).norm();
			if(r > EPSILON)
			{
				Real wr_ij = kernWR<Real>(r, smoothingLength);
				Coord g = -invAlpha_i*(pos_i - position[j])*wr_ij*(1.0f / r);
				vij = 0.5*(velBuf[pId] - velBuf[j]);
				dv[0] += vij[0] * g[0];
				dv[1] += vij[1] * g[1];
				dv[2] += vij[2] * g[2];
			}

		}
		crossVis[pId] = vb_cross_viscosity(visTop, visFloor, vb_trace_norm(dv[0], dv[1], dv[2]), K, N);
	}

	__global__ void VC_AttributeInit(DArray<Attribute> atts)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= atts.size()) return;
		atts[pId].setFluid();
		atts[pId].setDynamic();
	}

	template <typename Real>
	__global__ void VC_ViscosityValueUpdate(DArray<Real> viscosities, Real value)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= viscosities.size()) return;
		viscosities[pId] = value;
	}

	template<typename TDataType>
	ApproximateImplicitViscosity<TDataType>::~ApproximateImplicitViscosity()
	{
		m_alpha.clear();
		m_vy.clear();
		m_vr.clear();
		m_vp.clear();
		m_vpv.clear();
		m_VelocityReal.clear();
		velOld.clear();
		velBuf.clear();
		m_viscosity.clear();;
		if (m_reduce)
		{
			delete m_reduce;
		}
		if (m_arithmetic_v)
		{
			delete m_arithmetic_v;
		}
	};

	template<typename TDataType>
	bool ApproximateImplicitViscosity<TDataType>::SetCross()
	{
		CrossVisCeil = this->varViscosity()->getValue();
		CrossVisFloor = this->varLowerBoundViscosity()->getValue();
		CrossVisFloor = CrossVisFloor < CrossVisCeil ? CrossVisFloor : CrossVisCeil;

		Cross_K = this->varCrossK()->getValue();
		Cross_N = this->varCrossN()->getValue();
		std::cout << "*Non-Newtonian Fluid, Viscosity:" << CrossVisFloor << " to " << CrossVisCeil << ", K:"<< Cross_K << ", N:" << Cross_N << std::endl;
		return true;
	};

	template<typename TDataType>
	void ApproximateImplicitViscosity<TDataType>::constrain()
	{
		int num = this->inPosition()->size();

		if ((num != m_alpha.size())||
			(num != velOld.size()) ||
			(num != m_viscosity.size()))
		{
			m_alpha.resize(num);
			m_vy.resize(3 * num);
			m_vr.resize(3 * num);
			m_vp.resize(3 * num);
			m_vpv.resize(num);
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
		auto m_smoothingLength = this->inSmoothingLength()->getValue();
		auto m_restDensity = this->varRestDensity()->getValue();

		cuExecute(num, VC_ViscosityValueUpdate,	m_viscosity, this->varViscosity()->getValue());
	
		m_alpha.reset();
		cuExecute(num, VC_ComputeAlpha,
			m_alpha,
			m_position,
			m_attribute,
			m_neighborhood,
			m_smoothingLength);


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
		m_vy.reset();

		cuExecute(num, VC_Vis_AxComput,
				m_vy,
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

		m_vr.reset();
		cuExecute(num, VC_Vis_r_Comput,
				m_vr,
				m_vy,
				velOld,
				m_position,
				m_attribute
				);

		m_vp.assign(m_vr);

		Real Vrr = m_arithmetic_v->Dot(m_vr, m_vr);
		Real Verr = sqrt(Vrr / m_vr.size());
		int it = 0;
		Real initErr = Verr;

		while (it < 150 && Verr / initErr > 0.01f && Vrr > 1000.0f * EPSILON)
		{
			it++;
				cuExecute(num, VC_Vis_pToVector,
					m_vp,
					m_vpv,
					m_attribute
				);

				m_vy.reset();
				cuExecute(num, VC_Vis_AxComput,
					m_vy,
					m_vpv,
					m_position,
					m_alpha,
					m_attribute,
					m_neighborhood,
					m_restDensity,
					m_smoothingLength,
					dt,
					m_viscosity
					);

				Real alpha = Vrr / m_arithmetic_v->Dot(m_vp, m_vy);

				cuExecute(num, VC_Vis_CoordToReal,
					m_VelocityReal,
					m_velocity,
					m_attribute
					);

				Function2Pt::saxpy(m_VelocityReal, m_vp, m_VelocityReal, alpha);
				cuExecute(num, VC_Vis_RealToVeloctiy,
					m_velocity,
					m_VelocityReal,
					m_attribute
					);

				Function2Pt::saxpy(m_vr, m_vy, m_vr, -alpha);
				Real Vrr_old = Vrr;

				Vrr = m_arithmetic_v->Dot(m_vr, m_vr);
				Real beta = Vrr / Vrr_old;

				Function2Pt::saxpy(m_vp, m_vp, m_vr, beta);
				Verr = sqrt(Vrr / m_vr.size());
		}
	};

	DEFINE_CLASS(ApproximateImplicitViscosity);
}