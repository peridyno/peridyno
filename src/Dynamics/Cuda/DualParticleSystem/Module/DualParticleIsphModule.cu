#include <cuda_runtime.h>
#include "DualParticleIsphModule.h"
#include "Node.h"
#include "Field.h"
#include "ParticleSystem/Module/SummationDensity.h"
#include "Collision/Attribute.h"
#include "ParticleSystem/Module/Kernel.h"
#include "Algorithm/Function2Pt.h"
#include "Algorithm/CudaRand.h"


namespace dyno
{
	IMPLEMENT_TCLASS(DualParticleIsphModule, TDataType)

	template<typename TDataType>
	DualParticleIsphModule<TDataType>::DualParticleIsphModule()
		: ConstraintModule()
		, m_airPressure(Real(0))
		, m_reduce(NULL)
		, m_reduce_r(NULL)
		, m_arithmetic(NULL)
		, m_arithmetic_r(NULL)
	{
		this->inParticleAttribute()->tagOptional(true);
		this->inBoundaryNorm()->tagOptional(true);
		
		this->varSamplingDistance()->setValue(Real(0.005));
		this->varRestDensity()->setValue(Real(1000));
		this->varSmoothingLength()->setValue(Real(    0.0125));
		//this->varPpeSmoothingLength()->setValue(Real( 0.0125));

		m_summation = std::make_shared<SummationDensity<TDataType>>();
		this->varRestDensity()->connect(m_summation->varRestDensity());
		this->varSmoothingLength()->connect(m_summation->inSmoothingLength());
		this->varSamplingDistance()->connect(m_summation->inSamplingDistance());
		this->inRPosition()->connect(m_summation->inPosition());
		this->inNeighborIds()->connect(m_summation->inNeighborIds());

		m_vr_summation = std::make_shared<SummationDensity<TDataType>>();
		this->varRestDensity()->connect(m_vr_summation->varRestDensity());
		this->varSmoothingLength()->connect(m_vr_summation->inSmoothingLength());
		this->varSamplingDistance()->connect(m_vr_summation->inSamplingDistance());
		this->inVPosition()->connect(m_vr_summation->inPosition());
		this->inRPosition()->connect(m_vr_summation->inOther());
		this->inVRNeighborIds()->connect(m_vr_summation->inNeighborIds());

		m_vv_summation = std::make_shared<SummationDensity<TDataType>>();
		this->varRestDensity()->connect(m_vv_summation->varRestDensity());
		this->varSmoothingLength()->connect(m_vv_summation->inSmoothingLength());
		this->varSamplingDistance()->connect(m_vv_summation->inSamplingDistance());
		this->inVPosition()->connect(m_vv_summation->inPosition());
		this->inVVNeighborIds()->connect(m_vv_summation->inNeighborIds());

		//outfile_iter.open("DPISPH_ITER.txt");
		//outfile_virtualNumber.open("DPISPH_VIRTUAL.txt");
		//outfile_density.open("DPISPH_DENSITY.txt");
	}

	template<typename TDataType>
	DualParticleIsphModule<TDataType>::~DualParticleIsphModule()
	{

		m_r.clear();
		m_Aii.clear();
		m_virtualAirFlag.clear();
		m_pressure.clear();
		m_virtualVelocity.clear();
		m_source.clear();
		m_Ax.clear();
		m_r.clear();
		m_p.clear();
		m_Gp.clear();
		m_GpNearSolid.clear();
		m_RealPressure.clear();

		if (m_reduce)
		{
			delete m_reduce;
		}
		if (m_arithmetic)
		{
			delete m_arithmetic;
		}
		if (m_reduce_r)
		{
			delete m_reduce_r;
		}
		if (m_arithmetic_r)
		{
			delete m_arithmetic_r;
		}

		//m_RealVolumeEst.clear();
		//m_VirtualVolumeEst.clear();

		//if (outfile_iter.is_open())
		//{
		//	outfile_iter << '\n';
		//	outfile_iter.close();
		//}

		//if (outfile_virtualNumber.is_open())
		//{
		//	outfile_virtualNumber << '\n';
		//	outfile_virtualNumber.close();
		//}

		//if (outfile_density.is_open())
		//{
		//	outfile_density << '\n';
		//	outfile_density.close();
		//}
	}

	/*
	* MatrixDotVector
	*/
	template <typename Vec4, typename Matrix4x4>
	__device__ inline Vec4 MatrixDotVector(const Matrix4x4 M, const  Vec4 V)
	{
		return Vec4(
			V[0] * M(0, 0) + V[1] * M(0, 1) + V[2] * M(0, 2) + V[3] * M(0, 3),
			V[0] * M(1, 0) + V[1] * M(1, 1) + V[2] * M(1, 2) + V[3] * M(1, 3),
			V[0] * M(2, 0) + V[1] * M(2, 1) + V[2] * M(2, 2) + V[3] * M(2, 3),
			V[0] * M(3, 0) + V[1] * M(3, 1) + V[2] * M(3, 2) + V[3] * M(3, 3)
		);
	}

	/*
	* Name		:	UpdateVelocity;
	* Function	:	Update velocities with the pressure gradient field;
	* Formular	:	vi = vi + dt * Gp / rho;
	*/
	template <typename Real, typename Coord>
	__global__ void DualParticle_UpdateVelocity(
		DArray<Coord> gradientPress,
		DArray<Coord> velocity,
		DArray<Real> density,
		Real mass,
		Real dt
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= velocity.size()) return;

		Coord temp = dt * gradientPress[pId] / density[pId];
		velocity[pId] = velocity[pId] - temp;
	}


	template<typename TDataType>
	bool DualParticleIsphModule<TDataType>::virtualArraysResize()
	{
		int num = this->inRPosition()->size();
		int num_v = this->inVPosition()->size();

		if (m_pressure.size() != num_v)
			m_pressure.resize(num_v);
		if (m_Ax.size() != num_v)
			m_Ax.resize(num_v);
		if (m_Aii.size() != num_v)
			m_Aii.resize(num_v);
		if (m_r.size() != num_v)
			m_r.resize(num_v);
		if (m_p.size() != num_v)
			m_p.resize(num_v);
		if (m_virtualAirFlag.size() != num_v)
			m_virtualAirFlag.resize(num_v);
		if (m_source.size() != num_v)
			m_source.resize(num_v);
		if (m_virtualVelocity.size() != num_v)
			m_virtualVelocity.resize(num_v);
		if (this->outVirtualBool()->isEmpty())
			this->outVirtualBool()->allocate();
		if (this->outVirtualWeight()->isEmpty())
			this->outVirtualWeight()->allocate();
		if (this->outVirtualBool()->size() != num_v)
			this->outVirtualBool()->resize(num_v);
		if (this->outVirtualWeight()->size() != num_v)
			this->outVirtualWeight()->resize(num_v);
		//if (m_VirtualVolumeEst.size() != num_v)
		//	m_VirtualVolumeEst.resize(num_v);

		if (m_reduce)
		{
			delete m_reduce;
			m_reduce = Reduction<float>::Create(num_v);
		}
		else 
		{
			m_reduce = Reduction<float>::Create(num_v);
		}

		if (m_reduce_r)
		{
			delete m_reduce_r;
			m_reduce_r = Reduction<float>::Create(num);
		}
		else
		{
			m_reduce_r = Reduction<float>::Create(num);
		}

		if (m_arithmetic)
		{
			delete m_arithmetic;
			m_arithmetic = Arithmetic<float>::Create(num_v);
		}
		else
		{
			m_arithmetic = Arithmetic<float>::Create(num_v);
		}
		return true;
	}

	template<typename TDataType>
	bool DualParticleIsphModule<TDataType>::realArraysResize()
	{
		int num = this->inRPosition()->size();

		if (m_Gp.size() != num)
			m_Gp.resize(num);

		if (m_GpNearSolid.size() != num)
			m_GpNearSolid.resize(num);

		if (m_RealPressure.size() != num)
		{
			m_RealPressure.resize(num);
			m_RealPressure.reset();
		}

		//if (m_RealVolumeEst.size() != num)
		//{
		//	m_RealVolumeEst.resize(num);
		//}
		if (m_arithmetic_r)
		{
			delete m_arithmetic_r;
			m_arithmetic_r = Arithmetic<float>::Create(num);
		}
		else
		{
			m_arithmetic_r = Arithmetic<float>::Create(num);
		}
		return true;
	}



	/*
	* Name		: DualParticle_SolidVirtualParticleDetect
	* Function	: A virtual particle is Neumann boundary particle or not?
	*/
	template <typename Real, typename Coord>
	__global__ void  DualParticle_VirtualSolidParticleDetection(
		DArray<bool> solidVirtual,
		DArray<Real> fluidFraction,
		DArray<Coord> virtualPosition,
		DArray<Coord> realPosition,
		DArray<Attribute> attribute,
		DArray<Real> density,
		DArrayList<int> vr_neighbors,
		DArray<bool> virtualAir,
		CubicKernel<Real> kernel,
		Real mass,
		Real threshold,
		Real h
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= solidVirtual.size()) return;

		solidVirtual[pId] = false;
		List<int> & list_i_vr = vr_neighbors[pId];
		int nbSize = list_i_vr.size();
		Real c = 0.0f;
		Real w = 0.0f;
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j	= list_i_vr[ne];
			Real r_ij = (virtualPosition[pId] - realPosition[j]).norm();
			if (!attribute[j].isFluid())
			{
				c += kernel.Weight(r_ij, h) * mass / density[j];
			}
			w += kernel.Weight(r_ij, h) * mass / density[j];
		}

		if (w < EPSILON) w = EPSILON;
		fluidFraction[pId]	= c / w;

		if (fluidFraction[pId] > threshold)
		{
			solidVirtual[pId] = true;			
		}
		else
		{
			solidVirtual[pId] = false;
		}

	}


	template <typename Real, typename Coord>
	__global__ void  DualParticle_SolidBoundaryDivergenceCompsate(
		DArray<Real> source,
		DArray<bool> solidVirtual,
		DArray<bool> airVirtual,
		DArray<Coord> virtualVelocity,
		DArray<Coord> realVelocity,
		DArray<Coord> virtualPosition,
		DArray<Coord> realPosition,
		DArray<Attribute> attribute,
		DArray<Coord> norm,
		DArray<Real> density,
		DArrayList<int> vr_neighbors,
		CubicKernel<Real> kernel,
		Real restDensity,
		Real mass,
		Real dt,
		Real h
		)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= virtualPosition.size()) return;
		if (solidVirtual[pId] == true) return;
		if (airVirtual[pId] == true) return;

		List<int> & list_i_vr = vr_neighbors[pId];
		int nbSize_vr = list_i_vr.size();

		Real value(0);
		for (int ne = 0; ne < nbSize_vr; ne++)
		{
			int j = list_i_vr[ne];
			if (!attribute[j].isFluid())
			{

				Real r_ij = (virtualPosition[pId] - realPosition[j]).norm();
				Coord Gwij(0);

				if (r_ij > EPSILON && !attribute[j].isFluid())
				{

					Coord dVel = virtualVelocity[pId] - realVelocity[j];
					Real magNVel = dVel.dot(norm[j]);
					Coord nVel = magNVel * norm[j];
					Coord tVel = dVel - nVel;

					Gwij = kernel.Gradient(r_ij, h) * (virtualPosition[pId] - realPosition[j]) / r_ij;

					if (magNVel < -EPSILON)
					{
						value += 2 * (nVel + 0.01*tVel).dot(Gwij) * mass / density[pId];
					}
					else
					{
						value += 2 * (0.1 * nVel + 0.01*tVel).dot(Gwij) * mass / density[pId];
					}
				}
			}
		}
		
		source[pId] -= -value * restDensity / dt;
	}

	/*
	* @briedf Use CSPH method to calculate the virtual particle velocity;
	* Virtual-air particle velocity should be zero; 
	* 
	*/
	template <typename Real, typename Coord>
	__global__ void  DualParticle_SmoothVirtualVelocity(
		DArray<Coord> virtualVelocity,
		DArray<Coord> velocity,
		DArray<Coord> virtualPosition,
		DArray<Coord> realPosition,
		DArrayList<int> vr_neighbors,
		DArray<Attribute> r_att,
		DArray<bool> virtualAir,
		DArray<bool> virtualSolid,
		DArray<Real> density,
		CubicKernel<Real> kernel,
		Real mass,
		Real h
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= virtualPosition.size()) return;
		if (virtualSolid[pId] == true) return;
		if (virtualAir[pId] == true)
		{
			virtualVelocity[pId] = Coord(0.0f);
			return;
		}

		List<int>& list_i_vr = vr_neighbors[pId];
		int nbSize_vr = list_i_vr.size();
		Real total_w(0);
		Coord total_vw(0);
		for (int ne = 0; ne < nbSize_vr; ne++)
		{
			int j = list_i_vr[ne];
			if (r_att[j].isDynamic())
			{
				Real r_ij = (virtualPosition[pId] - realPosition[j]).norm();
				Real wij = kernel.Weight(r_ij, h);
				total_w += wij;
				total_vw += velocity[j] * wij;
			}
		}

		if (total_w > EPSILON)
		{
			virtualVelocity[pId] = total_vw / total_w;
		}
		else
		{
			virtualVelocity[pId] = Coord(0.0f);
		}
	}

	template <typename Real, typename Coord>
	__global__ void  DualParticle_SourceTerm(
		DArray<Real> source,
		DArray<Coord> virtualVelocity,
		DArray<Coord> velocity,
		DArray<Coord> virtualPosition,
		DArray<Coord> realPosition,
		DArray<Real> density,
		DArray<Real> vdensity,
		DArrayList<int> vr_neighbors,
		DArray<Attribute> attribute,
		DArray<bool> virtualAir,
		DArray<bool> virtualSolid,
		DArray<Coord> boundaryNorm,
		CubicKernel<Real> kernel,
		Real restDensity,
		Real mass,
		Real dt,
		Real h
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= source.size()) return;
		if (virtualSolid[pId] == true)
		{
			source[pId] = 0.0f;
			return;
		}

		if (virtualAir[pId] == true)
		{
			source[pId] = 0.0f;
		}
		else 
		{
			List<int> & list_i_vr = vr_neighbors[pId];
			int nbSize_vr = list_i_vr.size();

			Real value(0);
			for (int ne = 0; ne < nbSize_vr; ne++)
			{
				int j = list_i_vr[ne];
				Real r_ij = (virtualPosition[pId] - realPosition[j]).norm();
				Coord Gwij(0);
				if(r_ij > EPSILON & attribute[j].isFluid())
				{
					Gwij = kernel.Gradient(r_ij, h) * (virtualPosition[pId] - realPosition[j]) / r_ij;
				}
				value += (velocity[j] - virtualVelocity[pId]).dot(Gwij) * mass / vdensity[pId];
			}
			source[pId] = -value * restDensity / dt;
		}
	}

	/*
	* Name: VirtualAirParticleDetection
	* Return: Virtual-air particle flag
	* Function:  rho_v(virtual particles'real density) < threashold ? true: false   
	*/
	template <typename Real>
	__global__ void DualParticle_VirtualAirParticleDetection
	(
		DArray<bool> m_virtualAirFlag,
		DArrayList<int> vr_neighbors,
		DArray<Real> rho_v,
		Real threshold
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= m_virtualAirFlag.size()) return;

		List<int> & list_i_vr = vr_neighbors[pId];
		int nbSize_vr = list_i_vr.size();


		if ((rho_v[pId] > threshold) && (nbSize_vr > 0))
		{
			m_virtualAirFlag[pId] = false;

		}
		else
		{
			m_virtualAirFlag[pId] = true;
		}
	}

	template <typename Real, typename Coord>
	__global__ void DualParticle_LaplacianPressure
	(
		DArray<Real> Ax,
		DArray<Real> Aii,
		DArray<Real> pressure,
		DArray<Coord> v_pos,
		DArray<bool> virtualAir,
		DArray<bool> virtualSolid,
		DArrayList<int> vv_neighbors,
		DArray<Real> rho_vv,
		CubicKernel<Real> kernel,
		Real mass,
		Real h
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= Ax.size()) return;
		if (virtualSolid[pId] == true)
		{
			return;
		}

		if (virtualAir[pId] == true)
		{
			Ax[pId] = 0.0f;
			pressure[pId] = 0.0f;
			return;
		}
		List<int> & list_i = vv_neighbors[pId];
		int nbSize_i = list_i.size();

		Real temp = 0;
		for (int ne = 0; ne < nbSize_i; ne++)
		{
			int j = list_i[ne];

			Real rij = (v_pos[pId] - v_pos[j]).norm();

			if (rij > EPSILON && virtualAir[j] == false && virtualSolid[j] == false)
			{
				Real dwij = kernel.Gradient(rij, h) / rij;
				temp += 8 * mass * dwij / pow((rho_vv[pId] + rho_vv[j]), 2) * pressure[j];
			}

		}
		Ax[pId] = Aii[pId] * pressure[pId] + temp;
	}



	
	template <typename Real, typename Coord>
	__global__ void DualParticle_AiiInLaplacian
	(
		DArray<Real> Aii,
		DArray<Coord> v_pos,
		DArray<bool> virtualAir,
		DArray<bool> virtualSolid,
		DArrayList<int > vv_neighbors,
		DArray<Real> rho_vv,
		CubicKernel<Real> kernel,
		Real mass,
		Real h
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= Aii.size()) return;
		if (virtualSolid[pId] == true) return;
		if (virtualAir[pId] == true) { 
			Aii[pId] = EPSILON;
			return; 
		}
		List<int> & list_i = vv_neighbors[pId];
		int nbSize_i = list_i.size();

		Real temp = 0;

		for (int ne = 0; ne < nbSize_i; ne++)
		{
			int j = list_i[ne];

			Real rij = (v_pos[pId] - v_pos[j]).norm();

			if (rij > EPSILON && virtualAir[j] == false)
			{
				Real dwij = kernel.Gradient(rij, h) / rij;
				temp  += 8 * mass * dwij / pow((rho_vv[pId] + rho_vv[j]), 2);
			}
		}
		Aii[pId] = -temp;
	
	}

	/*
	* @brief: Using SPH aproach to calculate Gradient pressures of fluid particles.
	*/
	template <typename Real, typename Coord>
	__global__ void DualParticle_GradientPressure
	(
		DArray<Coord> gradient,
		DArray<Real> pressure,
		DArray<Coord> velocity,
		DArray<Coord> v_pos,
		DArray<Coord> r_pos,
		DArray<Attribute> attribute,
		DArray<bool> virtualAir,
		DArray<bool> virtualSolid,
		DArrayList<int> vr_neighbors,
		DArrayList<int> rv_neighbors,
		DArray<Real> rho_r,
		DArray<Real> rho_v,
		CubicKernel<Real> kernel,
		Real rho_0,
		Real dt,
		Real mass,
		Real h
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		
		if (pId >= gradient.size()) return;
		if (!attribute[pId].isFluid()) return;

		Coord value(0);
		List<int> & list_i_rv = rv_neighbors[pId];
		int nbSize_i = list_i_rv.size();

		for (int ne = 0; ne < nbSize_i; ne++)
		{
			int j = list_i_rv[ne];
			Real r_ij = (r_pos[pId] - v_pos[j]).norm();
			if ((r_ij > EPSILON) && (virtualAir[j] != true) && (virtualSolid[j] != true))
			{
				value += (mass / rho_0) * pressure[j] * (r_pos[pId] - v_pos[j]) * kernel.Gradient(r_ij, h) / r_ij;
			}
		}
		value = value * dt / rho_0;
		gradient[pId] = value;
		velocity[pId] -= gradient[pId] ;
	}


	/*
	* Gradient correction Martrix 
	* @Paper: Fang et al 2009. 
	* @Paper: Improved SPH methods for simulating free surface flows of viscous fluids
	* @Paper: Applied Numerical Mathematics 59 (2009) 251¨C271
	*/

	template <typename Real, typename Coord>
	__global__ void DualParticle_ImprovedGradient(
		DArray<Coord> improvedGradient,
		DArrayList<int> rv_neighbors,
		DArray<Coord> vpos,
		DArray<Coord> rpos,
		DArray<Real> vrho,
		DArray<bool> virtualAir,
		DArray<bool> virtualSolid,
		CubicKernel<Real> kernel,
		Real mass,
		Real rho_0,
		Real h
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= improvedGradient.size()) return;

		Real V = mass / rho_0;
		glm::mat4x4 C(0);
		List<int> & list_i_rr = rv_neighbors[pId];
		int nbSize_i = list_i_rr.size();
		for (int ne = 0; ne < nbSize_i; ne++)
		{
			int j = list_i_rr[ne];
			Coord xij = rpos[pId] - vpos[j];
			Real rij = xij.norm();
			Coord nij = xij / rij;
			Real weight = kernel.Weight(rij, h);
			Real dweight = kernel.Gradient(rij, h);
			C[0][0] +=			weight;
			C[0][1] += xij[0] * weight;
			C[0][2] += xij[1] * weight;
			C[0][3] += xij[2] * weight;

			C[1][0] +=			nij[0] * dweight;
			C[1][1] += xij[0] * nij[0] * dweight;
			C[1][2] += xij[1] * nij[0] * dweight;
			C[1][3] += xij[2] * nij[0] * dweight;

			C[2][0] +=			nij[1] * dweight;
			C[2][1] += xij[0] * nij[1] * dweight;
			C[2][2] += xij[1] * nij[1] * dweight;
			C[2][3] += xij[2] * nij[1] * dweight;

			C[3][0] +=			nij[2] * dweight;
			C[3][1] += xij[0] * nij[2] * dweight;
			C[3][2] += xij[1] * nij[2] * dweight;
			C[3][3] += xij[2] * nij[2] * dweight;

		}
		C = V * C;
		//glm::mat4x4 i;
		C = glm::inverse(C);
		if (pId == 132)
		{
			printf("GLM:INVERSE:");
			for (int ci = 0; ci < 4; ci++)
			{
				for (int cj = 0; cj < 4; cj++)
				{
					printf("%f, ", C[ci][cj]);
				}
				printf(" | ");
			}
			printf("\r\n");
		}
		
	}

	/*
	* @brief: Modify gradient pressures if the fluid particle near solids.
	* @Paper: Bridson. Fluid simulation for computer graphics, Second Edition. (Section 5.1, The Discrete Pressure Gradient)
	*/
	template <typename Real, typename Coord>
	__global__ void DualParticle_GradientNearSolid
	(
		DArray<Coord> gradidentComp,
		DArray<Coord> velocity,
		DArray<Coord> v_pos,
		DArray<Coord> r_pos,
		DArray<Attribute> attribute,
		DArray<Coord> norm,
		DArray<bool> virtualAir,
		DArray<bool> virtualSolid,
		DArrayList<int> rr_neighbors,
		DArray<Real> rho_r,
		DArray<Real> rho_v,
		CubicKernel<Real> kernel,
		Real rho_0,
		Real dt,
		Real mass,
		Real h
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= gradidentComp.size()) return;

		gradidentComp[pId] = Coord(0.0f);

		List<int> & list_i_rr = rr_neighbors[pId];
		int nbSize_i = list_i_rr.size();
		Coord &  vel_i = velocity[pId];
		Coord dvij(0.0f);
		for (int ne = 0; ne < nbSize_i; ne++)
		{
			int j = list_i_rr[ne];
			Real r_ij = (r_pos[pId] - r_pos[j]).norm();
			Real weight = kernel.Weight(r_ij, h);
			if (!attribute[j].isFluid())
			{

				Coord nij = (r_pos[pId] - r_pos[j]);
				if(nij.norm() > EPSILON)
				{
					nij = nij.normalize();
				}
				else
				{
					nij = Coord(1.0f, 0.0f, 0.0f);
				}

				Coord normal_j = norm[j];
				Coord dVel = vel_i - velocity[j];
				Real magNVel = dVel.dot(normal_j);
				Coord nVel = magNVel * normal_j;
				Coord tVel = dVel - nVel;
				if (magNVel < -EPSILON)
				{
					dvij += nij.dot(nVel +  0.01 * tVel) * weight * nij;
				}
				else
				{
					dvij += nij.dot(0.1 * nVel + 0.01 * tVel) * weight * nij;
				}

			}
		}
		gradidentComp[pId] = 2 * dt * dvij / rho_0;
		velocity[pId] -= gradidentComp[pId];
	}

	/*
	* @brief: Semi-analytical Dirichlet boundary.
	* @Paper: Nair and Tomar, Computers & Fluids, 2014, 10(102): 304-314, An improved free surface modeling for incompressible SPH.
	*/
	template <typename Real>
	__global__ void DualParticle_CorrectAii
	(
		DArray<Real> Aii,
		Real max_Aii,
		Real c
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= Aii.size()) return;
		max_Aii = max_Aii * c;
		Real temp = Aii[pId];
		if (temp < max_Aii)
		{
			temp = max_Aii;
		}
		Aii[pId] = temp;
	}

	/*
	* @brief: Modified density drifting error (Liu, et al. Tog 2024, Equation 13.)
	* @Paper: Abbas Khayyer and Hitoshi Gotoh. J. Comput. Phys. 230, 8 (2011). Enhancement of stability and accuracy of the moving particle semi-implicit method
	*/
	template <typename Real>
	__global__ void DualParticle_DensityCompensate
	(
		DArray<Real> source,
		DArray<Real> rho,
		DArray<bool> virtualAir,
		DArray<bool> virtualSolid,
		Real rho_0,
		Real dt,
		Real h
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= source.size()) return;
		if (virtualSolid[pId] == true)
		{
			return;
		}

		if (virtualAir[pId] == true)
		{
			return;
		}

		if (rho[pId] > rho_0)
		{
			source[pId] += 1000000.0f * (rho[pId] - rho_0) / rho_0;
		}
	}


	/*
	* @brief: Neumann Boundary in Matrix of pressrue Poisson eqtution.
	* @Paper: Bridson. Fluid simulation for computer graphics, Second Edition. (Section 5.3, The Pressure Equations.)
	*/
	template <typename Real, typename Coord>
	__global__ void DualParticle_AiiNeumannCorrect
	(
		DArray<Real> Aii,
		DArray<bool> virtualAir,
		DArray<bool> virtualSolid,
		DArrayList<int > vv_neighbors,
		DArray<Real> rho_vv,
		DArray<Coord> v_pos,
		CubicKernel<Real> kernel,
		Real max_Aii,
		Real mass,
		Real h
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= Aii.size()) return;
		if (virtualSolid[pId] == true) return;
		if (virtualAir[pId] == true) return;

		List<int> & list_i = vv_neighbors[pId];
		int nbSize_i = list_i.size();

		Real temp = 0;
		int solidCounter = 0;

		for (int ne = 0; ne < nbSize_i; ne++)
		{
			int j = list_i[ne];
			Real rij = (v_pos[pId] - v_pos[j]).norm();

			if (rij > EPSILON &&  virtualSolid[j] == true)
			{
				Real dwij = kernel.Gradient(rij, h) / rij;
				temp += 8 * mass * dwij / pow((rho_vv[pId] + rho_vv[j]), 2);
				solidCounter++;
			}
		}
		Real value = Aii[pId] + temp;
		Aii[pId] = value;
	}

	template <typename Coord>
	__global__ void DualParticle_SolidVelocityReset
	(
		DArray<Coord> velocity,
		DArray<Attribute> attribute
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= velocity.size()) return;
		
		if (!attribute[pId].isFluid())
		{
			velocity[pId] = Coord(0.0f);
		}
	}

	template <typename Real>
	__global__ void DualParticle_VolumeTest
	(
		DArray<Real> volume,
		DArray<Real> density,
		Real mass
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= volume.size()) return;
		volume[pId] = mass / density[pId];
	}


	template <typename Real>
	__global__ void DualParticle_VirtualVolumeTest
	(
		DArray<Real> volume,
		DArray<Real> density,
		DArray<bool> air,
		Real mass
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= volume.size()) return;
		if (air[pId] == false)
			volume[pId] = mass / density[pId];
		else
			volume[pId] = 0.0f;
	}

	template <typename Real, typename Coord>
	__global__ void DualParticle_PressureV2R
	(
		DArray<Real> RealPressure,
		DArray<Real> VirtualPressure,
		DArray<Coord> RPosition,
		DArray<Coord> VPosition,
		DArray<bool> virtualAir,
		DArray<bool> virtualSolid,
		DArrayList<int> RVNeighbors,
		Real mass,
		CubicKernel<Real> kernel,
		Real h
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= RPosition.size()) return;

		int nbSize_i = RVNeighbors[pId].size();
		List<int>& list_i = RVNeighbors[pId];

		Real totalWeight(0.0f);
		Real pressureWeight(0.0f);

		for (int ne = 0; ne < nbSize_i; ne++)
		{
			int j = list_i[ne];
			if ((virtualAir[j] != true)&&(virtualSolid[j] != true))
			{
				Real rij = (RPosition[pId] - VPosition[j]).norm();
				Real wij = kernel.Weight(rij, h);

				if (rij < EPSILON)
				{
					RealPressure[pId] = VirtualPressure[j];
					return;
				}

				totalWeight += wij;
				pressureWeight += VirtualPressure[j] * wij;
			}
		}
		if (totalWeight < 1000 * EPSILON) totalWeight = 1000 * EPSILON;

		Real avr_pressure = pressureWeight / totalWeight;
		RealPressure[pId] = avr_pressure;
	}


	template <typename Real, typename Coord>
	__global__ void DualParticle_PressureR2V
	(
		DArray<Real> VirtualPressure,
		DArray<Real> RealPressure,
		DArray<Coord> RPosition,
		DArray<Coord> VPosition,
		DArray<bool> virtualAir,
		DArray<bool> virtualSolid,
		DArrayList<int> VRNeighbors,
		Real mass,
		CubicKernel<Real> kernel,
		Real h
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= VirtualPressure.size()) return;
		if ((virtualAir[pId] == true)||(virtualSolid[pId] == true)) return;

		int nbSize_i = VRNeighbors[pId].size();
		List<int>& list_i = VRNeighbors[pId];

		Real totalWeight(0.0f);
		Real pressureWeight(0.0f);

		for (int ne = 0; ne < nbSize_i; ne++)
		{
			int j = list_i[ne];

			Real rij = (VPosition[pId] - RPosition[j]).norm();
			Real wij = kernel.Weight(rij, h);

			if ((rij < EPSILON))
			{
				VirtualPressure[pId] = RealPressure[j];
				return;
			}

			totalWeight += wij;
			pressureWeight += RealPressure[j] * wij;
		}

		if (totalWeight > 1000 * EPSILON)
		{
			Real avr_pressure = pressureWeight / totalWeight;
			VirtualPressure[pId] = avr_pressure;
		}
		else
		{
			VirtualPressure[pId] = 0.0f;
		}
	}


	/*
	* @brief: Using euleric finite divegence aproach to calculate Gradient pressures 
	*/
	template <typename Real, typename Coord>
	__global__ void DualParticle_VirtualGradientPressureFD(
		DArray<Coord> vGp,
		DArray<Coord> vpos,
		DArrayList<int> vv_neighbors,
		DArray<Real> pressure,
		DArray<bool> virtualAir,
		DArray<bool> virtualSolid,
		Real dx
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= vpos.size()) return;

		List<int> & list_i = vv_neighbors[pId];
		int nbSize_i = list_i.size();
		int count = 0;
		Coord gp(0.0f);
		for (int ne = 0; ne < nbSize_i; ne++)
		{
			int j = list_i[ne];
			Real rij = (vpos[pId] - vpos[j]).norm();
			if ((rij < dx)&&(rij > EPSILON))
			{
				count++;
				gp += (pressure[pId] - pressure[j]) * (vpos[pId] - vpos[j]) / rij;
			}
		}
		vGp[pId] = gp;
	}

	template <typename Real, typename Coord>
	__global__ void DualParticle_VirtualGradientPressureMeshless(
		DArray<Coord> vGp,
		DArray<Coord> vpos,
		DArrayList<int> vv_neighbors,
		DArray<Real> pressure,
		DArray<bool> virtualAir,
		DArray<bool> virtualSolid,
		CubicKernel<Real> kernel,
		Real mass,
		Real rho_0,
		Real h
	) {
	
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= vpos.size()) return;

		if (virtualAir[pId] == true)
		{
			vGp[pId] = Coord(0.0f);
			return;
		}

		List<int> & list_i = vv_neighbors[pId];
		int nbSize_i = list_i.size();
		Real dwij(0.0f);
		Coord value(0.0f);
		for (int ne = 0; ne < nbSize_i; ne++)
		{
			int j = list_i[ne];
			Real rij = (vpos[pId] - vpos[j]).norm();
			if(rij > EPSILON)
			{
				dwij = kernel.Gradient(rij, h);
				value += mass * (pressure[j] - pressure[pId]) * dwij * (vpos[pId] - vpos[j]) / (rij * rho_0);
			}
		}
		vGp[pId] = value;
	}

	/*
	* @brief: Using moving least square aproach to calculate Gradient pressures
	*/
	template <typename Real, typename Coord, typename Vector4, typename Matrix4x4>
	__global__ void DualParticle_MlsGradientPressure(
		DArray<Coord> rGp,
		DArray<Coord> velocity,
		DArray<Coord> vGp,
		DArray<Coord> vpos,
		DArray<Coord> rpos,
		DArrayList<int> rv_neighbors,
		DArray<Real> pressure,
		CubicKernel<Real> kernel,
		Real rho_0,
		Real dt,
		Real h
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= velocity.size()) return;

		List<int> & list_i = rv_neighbors[pId];
		int nbSize_i = list_i.size();

		Vector4 P(0.0f);		/* Basis Function: P((xi - xj)/h) */
		Vector4 P0(0.0f);		/* Basis Function: P(0) */
		Matrix4x4 M(0.0f);		/* Momentum Matrix of MLS*/
		Real dx_Ji(0.0f);		/* (xi - xj)/h*/
		Real dy_Ji(0.0f);		/* (yi - yj)/h*/
		Real dz_Ji(0.0f);		/* (zi - zj)/h*/
		Real wij(0.0f);			/*Weight*/
		Real rij(0.0f);
		
		/*Momentum Matrix*/
		for (int ne = 0; ne < nbSize_i; ne++)
		{
			int j = list_i[ne];
			rij = (rpos[pId] - vpos[j]).norm();
			wij = kernel.Weight(rij, h);

			dx_Ji = (vpos[j] - rpos[pId])[0] / h;
			dy_Ji = (vpos[j] - rpos[pId])[1] / h;
			dz_Ji = (vpos[j] - rpos[pId])[2] / h;
			
			M(0, 0) += wij;				M(0, 1) += wij * dx_Ji;					M(0, 2) += wij * dy_Ji;					M(0, 3) += wij * dz_Ji;
			M(1, 0) += wij * dx_Ji;		M(1, 1) += wij * dx_Ji * dx_Ji;			M(1, 2) += wij * dx_Ji * dy_Ji;			M(1, 3) += wij * dx_Ji * dz_Ji;
			M(2, 0) += wij * dy_Ji;		M(2, 1) += wij * dx_Ji * dy_Ji;			M(2, 2) += wij * dy_Ji * dy_Ji;			M(2, 3) += wij * dy_Ji * dz_Ji;
			M(3, 0) += wij * dz_Ji;		M(3, 1) += wij * dx_Ji * dz_Ji;			M(3, 2) += wij * dy_Ji * dz_Ji;			M(3, 3) += wij * dz_Ji * dz_Ji;
		}

		Matrix4x4 M_inv = M.inverse();
		
		Coord tgp(0.0f);
		for (int ne = 0; ne < nbSize_i; ne++)
		{
			int j = list_i[ne];
			rij = (rpos[pId] - vpos[j]).norm();
			wij = kernel.Weight(rij, h);

			dx_Ji = (vpos[j] - rpos[pId])[0] / h;
			dy_Ji = (vpos[j] - rpos[pId])[1] / h;
			dz_Ji = (vpos[j] - rpos[pId])[2] / h;

			P[0] = 1.0f;
			P[1] = dx_Ji;
			P[2] = dy_Ji;
			P[3] = dz_Ji;

			P0[0] = 1.0f;
			P0[1] = 0.0f;
			P0[2] = 0.0f;
			P0[3] = 0.0f;

			tgp += P0.dot(MatrixDotVector(M_inv, P)) * wij * vGp[j];
			
		}

		rGp[pId] = dt * tgp / rho_0;

		velocity[pId] -= rGp[pId];

		}


	__global__ void DualParticle_AttributeInit
	(
		DArray<Attribute> atts
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= atts.size()) return;

		atts[pId].setFluid();
		atts[pId].setDynamic();
	}

	template<typename TDataType>
	void DualParticleIsphModule<TDataType>::constrain()
	{
		int num = this->inRPosition()->size();
		int num_v = this->inVPosition()->size();

		cudaDeviceSynchronize();

		if (m_Gp.size() != num)
		{
			realArraysResize();
		}
		if (m_Ax.size() != num_v)
		{
			virtualArraysResize();
		}

		auto & m_virtualSolidFlag = this->outVirtualBool()->getData();
		Real dt = this->inTimeStep()->getData();
		Real h = this->varSmoothingLength()->getData();

		if (this->inParticleAttribute()->isEmpty() 
			|| this->inParticleAttribute()->size() != num
			|| this->inBoundaryNorm()->size() != num
			)
		{
			this->inParticleAttribute()->allocate();
			this->inParticleAttribute()->resize(num);
			cuExecute(num, DualParticle_AttributeInit, this->inParticleAttribute()->getData());
			this->inBoundaryNorm()->resize(num);
			this->inBoundaryNorm()->reset();
		}

		m_summation->update();
		m_vv_summation->update();
		m_vr_summation->update();

		m_particleMass = m_summation->getParticleMass();
		m_v_particleMass = m_vv_summation->getParticleMass();

		Real restDensity= this->varRestDensity()->getValue();

		Real MaxVirtualDensity = 
			m_reduce->maximum(
				m_vr_summation->outDensity()->getData().begin(), 
				m_vr_summation->outDensity()->getData().size()
			);

		cuExecute(num_v, DualParticle_VirtualAirParticleDetection,
			m_virtualAirFlag,
			this->inVRNeighborIds()->getData(),
			m_vr_summation->outDensity()->getData(),
			MaxVirtualDensity * 0.1f
		);

		cuExecute(num_v, DualParticle_VirtualSolidParticleDetection,
			m_virtualSolidFlag,
			this->outVirtualWeight()->getData(),
			this->inVPosition()->getData(),
			this->inRPosition()->getData(),
			this->inParticleAttribute()->getData(),
			m_summation->outDensity()->getData(),
			this->inVRNeighborIds()->getData(),
			m_virtualAirFlag,
			kernel,
			m_particleMass,
			0.999f,
			h
		);
		
		cuExecute(num_v, DualParticle_SmoothVirtualVelocity,
			m_virtualVelocity,
			this->inVelocity()->getData(),
			this->inVPosition()->getData(),
			this->inRPosition()->getData(),
			this->inVRNeighborIds()->getData(),
			this->inParticleAttribute()->getData(),
			m_virtualAirFlag,
			m_virtualSolidFlag,
			m_summation->outDensity()->getData(),
			kernel,
			m_particleMass,
			h
		);

		m_source.reset();

		cuExecute(num, DualParticle_SolidVelocityReset,
			this->inVelocity()->getData(),
			this->inParticleAttribute()->getData()
		);

		cuExecute(num_v, DualParticle_SourceTerm,
			m_source,
			m_virtualVelocity,
			this->inVelocity()->getData(),
			this->inVPosition()->getData(),
			this->inRPosition()->getData(),
			m_summation->outDensity()->getData(),
			m_vr_summation->outDensity()->getData(),
			this->inVRNeighborIds()->getData(),
			this->inParticleAttribute()->getData(),
			m_virtualAirFlag,
			m_virtualSolidFlag,
			this->inBoundaryNorm()->getData(),
			kernel,
			restDensity,
			m_particleMass,		
			dt,				
			h			
		);

		cuExecute(num_v, DualParticle_SolidBoundaryDivergenceCompsate,
			m_source,				
			m_virtualSolidFlag,
			m_virtualAirFlag,
			m_virtualVelocity,			
			this->inVelocity()->getData(),		
			this->inVPosition()->getData(),			
			this->inRPosition()->getData(),			
			this->inParticleAttribute()->getData(),			
			this->inBoundaryNorm()->getData(),
			m_vr_summation->outDensity()->getData(),		
			this->inVRNeighborIds()->getData(),				
			kernel,					
			restDensity,
			m_v_particleMass,				
			dt,							
			h					
			);

		cuExecute(num_v, DualParticle_DensityCompensate,
				m_source,
				m_vr_summation->outDensity()->getData(),
				m_virtualAirFlag,
				m_virtualSolidFlag,
				restDensity,
				dt,
				h
				);

		m_r.reset();
		m_Ax.reset();

		m_pressure.reset();

		if (this->varWarmStart()->getValue())
		{
			cuExecute(num_v, DualParticle_PressureR2V,
				m_pressure,
				m_RealPressure,
				this->inRPosition()->getData(),
				this->inVPosition()->getData(),
				m_virtualAirFlag,
				m_virtualSolidFlag,
				this->inVRNeighborIds()->getData(),
				m_v_particleMass,
				kernel,
				h
			);
		}
		//else {
		//	m_pressure.reset();
		//}

			
		cuExecute(num_v, DualParticle_AiiInLaplacian,
			m_Aii,
			this->inVPosition()->getData(),
			m_virtualAirFlag,
			m_virtualSolidFlag,
			this->inVVNeighborIds()->getData(),
			m_vv_summation->outDensity()->getData(),
			kernel,
			m_v_particleMass,
			h
		);

		if ((frag_number <= 3) || abs(max_Aii) < EPSILON)
		{
			//if(m_reduce->maximum(m_Aii.begin(), m_Aii.size()) > 0)
			max_Aii = m_reduce->maximum(m_Aii.begin(), m_Aii.size());
		}

		cuExecute(num_v, DualParticle_CorrectAii,
			m_Aii,
			max_Aii,
			1.0f
		);

		cuExecute(num_v, DualParticle_AiiNeumannCorrect,
			m_Aii, 
			m_virtualAirFlag, 
			m_virtualSolidFlag,
			this->inVVNeighborIds()->getData(),  
			m_vv_summation->outDensity()->getData(), 
			this->inVPosition()->getData(),
			kernel, 
			max_Aii, 
			m_v_particleMass, 
			h
			);

		cuExecute(num_v, DualParticle_LaplacianPressure,
			m_Ax,
			m_Aii,
			m_pressure,
			this->inVPosition()->getData(),
			m_virtualAirFlag,
			m_virtualSolidFlag,
			this->inVVNeighborIds()->getData(),
			m_vv_summation->outDensity()->getData(),
			kernel,
			m_v_particleMass,
			h
			);

		Function2Pt::subtract(m_r, m_source, m_Ax);
		m_p.assign(m_r);

		Real rr = m_arithmetic->Dot(m_r, m_r);
		Real err = m_r.size() > 0 ? sqrt(rr / m_r.size()) : 0.0f;
		Real max_err = err;
		if (abs(max_err) < EPSILON) max_err = EPSILON;
		unsigned int iter = 0;
		Real threshold = this->varResidualThreshold()->getValue();
		while ((iter < 500) && (err / max_err > threshold) && (err > threshold))
		{
			iter++;

			m_Ax.reset();

			cuExecute(num_v, DualParticle_LaplacianPressure,
				m_Ax,
				m_Aii,
				m_p,
				this->inVPosition()->getData(),
				m_virtualAirFlag,
				m_virtualSolidFlag,
				this->inVVNeighborIds()->getData(),
				m_vv_summation->outDensity()->getData(),
				kernel,
				m_v_particleMass,
				h
				);

			float alpha = rr / m_arithmetic->Dot(m_p, m_Ax);
			Function2Pt::saxpy(m_pressure, m_p, m_pressure, alpha);
			Function2Pt::saxpy(m_r, m_Ax, m_r, -alpha);

			Real rr_old = rr;

			rr = m_arithmetic->Dot(m_r, m_r);

			Real beta = rr / rr_old;
			Function2Pt::saxpy(m_p, m_p, m_r, beta);
			err = sqrt(rr / m_r.size());
			//std::cout<<"*DUAL-ISPH:: iter:"<< iter <<": Err-" << err << std::endl;
		}
		std::cout << "*DUAL-ISPH::Solver::Iteration:" << iter << "||RelativeError:" << err/ max_err * 100 <<"%" << std::endl;

		//outfile_iter << iter;
		//outfile_iter << std::endl;
		//outfile_virtualNumber << this->inVPosition()->size();
		//outfile_virtualNumber << std::endl;

		m_GpNearSolid.reset();

		cuExecute(num, DualParticle_GradientPressure,
				m_Gp,
				m_pressure,
				this->inVelocity()->getData(),
				this->inVPosition()->getData(),
				this->inRPosition()->getData(),
				this->inParticleAttribute()->getData(),
				m_virtualAirFlag, 
				m_virtualSolidFlag,
				this->inVRNeighborIds()->getData(),
				this->inRVNeighborIds()->getData(),
				m_summation->outDensity()->getData(),
				m_vr_summation->outDensity()->getData(),
				kernel,
				restDensity,
				dt,
				m_v_particleMass,
				h
				);

		cuExecute(num, DualParticle_GradientNearSolid,
				m_GpNearSolid, 
				this->inVelocity()->getData(), 
				this->inVPosition()->getData(), 
				this->inRPosition()->getData(), 
				this->inParticleAttribute()->getData(), 
				this->inBoundaryNorm()->getData(),
				m_virtualAirFlag, 
				m_virtualSolidFlag, 
				this->inNeighborIds()->getData(), 
				m_summation->outDensity()->getData(),
				m_vr_summation->outDensity()->getData(), 
				kernel,
				restDensity,
				dt, 
				m_v_particleMass, 
				h
				);

		if (this->varWarmStart()->getValue())
		{
			m_RealPressure.reset();
			cuExecute(num, DualParticle_PressureV2R,
				m_RealPressure, 
				m_pressure, 
				this->inRPosition()->getData(), 
				this->inVPosition()->getData(), 
				m_virtualAirFlag,
				m_virtualSolidFlag,
				this->inRVNeighborIds()->getData(),
				m_v_particleMass, 
				kernel,
				h
			);
		}

		/*
		*  Gradient Pressure On Virtual Points (Finite Difference)
		*/
		//DualParticle_VirtualGradientPressureFD << <pDims_r, BLOCK_SIZE >> > (
		//	m_vGp,
		//	this->inVPosition()->getData(), //DArray<Coord> vpos,
		//	this->inVVNeighborIds()->getData(), //DArrayList<int> vv_neighbors,
		//	m_pressure, //DArray<Real> pressure,
		//	m_virtualAirFlag,	//DArray<bool> virtualAir,
		//	m_virtualSolidFlag,		//DArray<bool> virtualSolid,
		//	0.0051f
		//);


		/*
		*  Improved method for Gradient Pressure On Virtual Points 
		*/
		//DualParticle_ImprovedGradient << <pDims_r, BLOCK_SIZE >> > (
		//	m_improvedGradient,
		//	this->inRVNeighborIds()->getData(),
		//	this->inVPosition()->getData(),
		//	this->inRPosition()->getData(),
		//	m_vv_summation->outDensity()->getData(),
		//	m_virtualAirFlag,
		//	m_virtualSolidFlag,
		//	kernel,
		//	m_v_particleMass,
		//	1000.0f,
		//	h1
		//	);


		/*
		*  Gradient Pressure On Virtual Points (SPH)
		*/
		//DualParticle_VirtualGradientPressureMeshless << <pDims_v, BLOCK_SIZE >> > (
		//	m_vGp,
		//	this->inVPosition()->getData(), //DArray<Coord> vpos,
		//	this->inVVNeighborIds()->getData(), //DArrayList<int> vv_neighbors,
		//	//pseudoPressure, 
		//	m_pressure, ////DArray<Real> pressure,
		//	m_virtualAirFlag,	//DArray<bool> virtualAir,
		//	m_virtualSolidFlag,		//DArray<bool> virtualSolid,
		//	kernel,
		//	m_v_particleMass,
		//	1000.0f,
		//	0.011f
		//	);

		/*
		*  Moving least square approgh for Gradient Pressure On Virtual Points (SPH)
		*/
		//DualParticle_MlsGradientPressure <Real, Coord, Vector<Real, 4>, SquareMatrix<Real , 4>> << <pDims_r, BLOCK_SIZE >> >	(
		//		m_rGp,	//DArray<Coord> rGp,
		//		this->inVelocity()->getData(),
		//		m_vGp,	//DArray<Coord> vGp,
		//		this->inVPosition()->getData(), //DArray<Coord> vpos,
		//		this->inRPosition()->getData(), //DArray<Coord> rpos,
		//		this->inRVNeighborIds()->getData(),	//DArrayList<int> rv_neighbors,
		//		m_pressure,	//DArray<Real> pressure,
		//		kernel,
		//		1000.0f,
		//		dt,
		//		0.011	//Real h
		//	);


		//cuExecute(num, DualParticle_VolumeTest,
		//	m_RealVolumeEst,
		//	m_summation->outDensity()->getData(),
		//	m_v_particleMass
		//);

		//cuExecute(num_v, DualParticle_VolumeTest,
		//	m_VirtualVolumeEst,
		//	m_vr_summation->outDensity()->getData(),
		//	m_v_particleMass
		//);

		//Real VirtualVolumeEst = m_reduce->average(
		//	m_VirtualVolumeEst.begin(),
		//	m_VirtualVolumeEst.size()
		//);

		//Real RealVolumeEst = m_reduce_r->average(
		//	m_RealVolumeEst.begin(),
		//	m_RealVolumeEst.size()
		//);

		Real VirtualDensityEst = m_reduce->average(
			m_vr_summation->outDensity()->getData().begin(),
			m_vr_summation->outDensity()->getData().size()
		);

		Real RealDensityEst = m_reduce_r->average(
			m_summation->outDensity()->getData().begin(),
			m_summation->outDensity()->getData().size()
		);

		//outfile_density << VirtualDensityEst << "\t" << RealDensityEst << std::endl;
	}

	template<typename TDataType>
	bool DualParticleIsphModule<TDataType>::initializeImpl()
	{
		cuSynchronize();

		int num = this->inRPosition()->size();
		int num_v = this->inVPosition()->size();

		m_reduce = Reduction<float>::Create(num_v);
		m_reduce_r = Reduction<float>::Create(num);
		m_arithmetic = Arithmetic<float>::Create(num_v);
		m_arithmetic_r = Arithmetic<float>::Create(num);

		return true;
	}

	DEFINE_CLASS(DualParticleIsphModule);
}