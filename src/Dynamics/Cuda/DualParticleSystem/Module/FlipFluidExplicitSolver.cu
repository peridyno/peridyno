#include "FlipFluidExplicitSolver.h"
#include "Node.h"
#include "./ParticleSystem/Module/SummationDensity.h"

namespace dyno
{


	template<typename TDataType>
	FlipFluidExplicitSolver<TDataType>::FlipFluidExplicitSolver()
		: ConstraintModule()
	{
		//m_summation = std::make_shared<SummationDensity<TDataType>>();
		//this->varRestDensity()->connect(m_summation->varRestDensity());
		//this->varSmoothingLength()->connect(m_summation->inSmoothingLength());
		//this->varSamplingDistance()->connect(m_summation->inSamplingDistance());
		//this->inParticlePosition()->connect(m_summation->inPosition());
		//this->inParticleNeighborIds()->connect(m_summation->inNeighborIds());
		this->varInterpolationModel()->getDataPtr()->setCurrentKey(2);
	}

	template<typename TDataType>
	FlipFluidExplicitSolver<TDataType>::~FlipFluidExplicitSolver()
	{
		m_gridMass.clear();
		m_C.clear();
		m_J.clear();
		m_pVelo_old.clear();	//Old Particle Velocity 
		m_gVelo_old.clear();	//Old Grid Velocity
		m_FlipVelocity.clear(); //FLIP Velocity
	}

	template <typename Real>
	__device__ inline Real ExplicitFlip_WeightFunc(Real r)
	{
		r = r > 0.0f ? r : -r;
		if (r < 0.5f)
			return 0.75f - r * r;
		else if ((r >= 0.5f) && (r < 1.5f))
			return 0.5f * (1.5f - r) * (1.5f - r);
		else
			return 0.0f;
	}


	template <typename Coord, typename Matrix>
	__device__ inline Matrix ExplicitFlip_outerProduct(Coord A, Coord B)
	{

		return Matrix::SquareMatrix(
			A[0] * B[0], A[0] * B[1], A[0] * B[2],
			A[1] * B[0], A[1] * B[1], A[1] * B[2],
			A[2] * B[0], A[2] * B[1], A[2] * B[2]
		);
	}

	template <typename Real, typename Coord>
	__global__ void ExplicitFlip_GridReset(
		DArray<Coord> gridVelocity,
		DArray<Coord> gVelo_old,
		DArray<Real> gridMass
	)
	{
		int gId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (gId >= gridVelocity.size()) return;

		gridVelocity[gId] = Coord(0.0f);
		gVelo_old[gId] = Coord(0.0f);
		gridMass[gId] = 0.0f;
	}



	template <typename Real, typename Matrix>
	__global__ void ExplicitFlip_ParticleReset(
		DArray<Matrix> C,
		DArray<Real> J
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= C.size()) return;

		//C[pId] = Matrix::identityMatrix();
		C[pId] = Matrix::SquareMatrix(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
		J[pId] = 1.0f;
	}



	template <typename Real, typename Coord, typename Matrix>
	__global__ void ExplicitFlip_ParticleToGrid(
		DArray<Coord> gridVelocity,
		DArrayList<int> vr_neighbors,
		DArray<Real> gridMass,
		DArray<Coord> particlePosition,
		DArray<Coord> gridPosition,
		DArray<Coord> particleVelocity,
		DArray<Coord> FlipVelocity,
		DArray<Coord> gVelo_old,
		DArray<Coord> pVelo_old,
		DArray<Real> J,
		DArray<Matrix> C,
		Real dx,		
		Real dt,
		Real E,
		Real particleVolume,
		Real particleMass,
		uint Model
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= particlePosition.size()) return;

		List<int>& list_i = vr_neighbors[pId];
		int nbSize_i = list_i.size();

		Real stress = -dt * 4 * E * particleVolume * (J[pId] - 1) / (dx * dx);
		Matrix affine = Matrix::identityMatrix() * stress + particleMass * C[pId];
		Coord q = Coord(0.0f);

		particleVelocity[pId] += FlipVelocity[pId];
		pVelo_old[pId] = particleVelocity[pId];

		for (int ne = 0; ne < nbSize_i; ne++)
		{
			int j = list_i[ne];
			Coord dxij = (particlePosition[pId] - gridPosition[j]);

			q[0] = abs(dxij[0] / dx);
			q[1] = abs(dxij[1] / dx);
			q[2] = abs(dxij[2] / dx);
			Real weight = ExplicitFlip_WeightFunc(q[0]) * ExplicitFlip_WeightFunc(q[1]) * ExplicitFlip_WeightFunc(q[2]);

			if (weight > EPSILON)
			{
				atomicAdd(&gridVelocity[j][0],
					weight * (particleMass * particleVelocity[pId] + affine * dxij)[0]);
				atomicAdd(&gridVelocity[j][1],
					weight * (particleMass * particleVelocity[pId] + affine * dxij)[1]);
				atomicAdd(&gridVelocity[j][2],
					weight * (particleMass * particleVelocity[pId] + affine * dxij)[2]);


				atomicAdd(&gVelo_old[j][0],
					weight * particleMass * particleVelocity[pId][0]);
				atomicAdd(&gVelo_old[j][1],
					weight * particleMass * particleVelocity[pId][1]);
				atomicAdd(&gVelo_old[j][2],
					weight * particleMass * particleVelocity[pId][2]);


				atomicAdd(&gridMass[j],
					weight * particleMass);
			}
		}
	}

	template <typename Real, typename Coord>
	__global__ void ExplicitFlip_GridVelocityUpdate(
		DArray<Coord> gridVelocity,
		DArray<Real> gridMass,
		DArray<Coord> Velocity_old
	)
	{
		int gId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (gId >= gridVelocity.size()) return;

		if (gridMass[gId] > 0.0f) {
			Velocity_old[gId] = Velocity_old[gId] / gridMass[gId];
			gridVelocity[gId] = gridVelocity[gId] / gridMass[gId];
		}


	}


	template <typename Real, typename Coord, typename Matrix>
	__global__ void ExplicitFlip_GridToParticle(
		DArray<Coord> particlePosition,
		DArrayList<int> vr_neighbors,
		DArray<Coord> gridPosition,
		DArray<Coord> particleVelocity,
		DArray<Coord> gridVelocity,
		DArray<Coord> pVelo_old,
		DArray<Coord> gVelo_old,
		DArray<Coord> FlipVelocity,
		DArray<Real> gridMass,
		DArray<Real> J,
		DArray<Matrix> C,
		Real dx,		//grid spacing
		Real dt,
		Real E,
		Real particleVolume,
		Real particleMass,
		Real FlipAlpha,
		uint Model
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= particleVelocity.size()) return;

		Coord new_v = Coord(0.0f, 0.0f, 0.0f);
		Matrix new_C = Matrix::SquareMatrix(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
		Coord q = Coord(0.0f);
		Coord flipVelo_old = Coord(0.0f);

		List<int>& list_i = vr_neighbors[pId];
		int nbSize_i = list_i.size();

		for (int ne = 0; ne < nbSize_i; ne++)
		{
			int j = list_i[ne];
			Coord dxij = (particlePosition[pId] - gridPosition[j]);


			q[0] = abs(dxij[0] / dx);
			q[1] = abs(dxij[1] / dx);
			q[2] = abs(dxij[2] / dx);
			Real weight = ExplicitFlip_WeightFunc(q[0]) * ExplicitFlip_WeightFunc(q[1]) * ExplicitFlip_WeightFunc(q[2]);

			Coord g_v = gridVelocity[j];

			new_v += weight * g_v;

			new_C += 4 * weight * ExplicitFlip_outerProduct<Coord, Matrix>(g_v, dxij) / (dx * dx);

			flipVelo_old += weight * gVelo_old[j];
		}

		FlipVelocity[pId] = FlipAlpha * (pVelo_old[pId] - flipVelo_old);

		//C[pId] = new_C;

		particleVelocity[pId] = new_v;

		J[pId] *= 1 + dt * new_C.trace();

	}



	template<typename TDataType>
	void FlipFluidExplicitSolver<TDataType>::constrain()
	{

		std::cout << "FlipFluidExplicitSolver" << std::endl;

		int p_num = this->inParticlePosition()->size();
		int g_num = this->inAdaptGridPosition()->size();
	
		if (this->inGridVelocity()->size() != g_num) {
			this->inGridVelocity()->resize(g_num);
		}

		if (m_gridMass.size() != g_num)
			m_gridMass.resize(g_num);

		if (m_C.size() != p_num)
			m_C.resize(p_num);

		if (m_J.size() != p_num)
			m_J.resize(p_num);

		if (m_pVelo_old.size() != p_num)
			m_pVelo_old.resize(p_num);

		if (m_gVelo_old.size() != g_num)
			m_gVelo_old.resize(g_num);

		if (m_FlipVelocity.size() != p_num)
			m_FlipVelocity.resize(p_num);

		if (this->inFrameNumber()->getValue() == 0)
		{
			cuExecute(p_num, ExplicitFlip_ParticleReset, m_C, m_J);
			m_FlipVelocity.reset();
		}

		Real particleSpacing = this->inSamplingDistance()->getData();
		particle_Volume = 0.25 * particleSpacing * particleSpacing * particleSpacing;
		particle_Mass = particle_Volume * particle_Density;

		cuExecute(g_num, ExplicitFlip_GridReset,
			this->inGridVelocity()->getData(),
			m_gVelo_old,
			m_gridMass
		);

		cuExecute(p_num, ExplicitFlip_ParticleToGrid,
			this->inGridVelocity()->getData(),
			this->inPGNeighborIds()->getData(),
			m_gridMass,
			this->inParticlePosition()->getData(),
			this->inAdaptGridPosition()->getData(),
			this->inParticleVelocity()->getData(),
			m_FlipVelocity,
			m_gVelo_old,
			m_pVelo_old,
			m_J,
			m_C,
			this->inGridSpacing()->getData(),
			this->inTimeStep()->getData(),
			this->varStiffness()->getValue(),
			particle_Volume,
			particle_Mass,
			(uint)(this->varInterpolationModel()->getDataPtr()->currentKey())
		);


		cuExecute(g_num, ExplicitFlip_GridVelocityUpdate,
			this->inGridVelocity()->getData(),
			m_gridMass,
			m_gVelo_old
		);


		cuExecute(p_num, ExplicitFlip_GridToParticle,
			this->inParticlePosition()->getData(),
			this->inPGNeighborIds()->getData(),
			this->inAdaptGridPosition()->getData(),
			this->inParticleVelocity()->getData(),
			this->inGridVelocity()->getData(),
			m_pVelo_old,
			m_gVelo_old,
			m_FlipVelocity,
			m_gridMass,
			m_J,  
			m_C,
			this->inGridSpacing()->getData(),
			this->inTimeStep()->getData(),
			this->varStiffness()->getValue(),
			particle_Volume,
			particle_Mass,
			this->varFlipAlpha()->getData(),
			(uint)(this->varInterpolationModel()->getDataPtr()->currentKey())
		);

	}

	DEFINE_CLASS(FlipFluidExplicitSolver);
}