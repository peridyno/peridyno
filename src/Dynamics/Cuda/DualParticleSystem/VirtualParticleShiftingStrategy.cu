#include "VirtualParticleShiftingStrategy.h"

#include "Node.h"
#include "ParticleSystem/Module/SummationDensity.h"
#include "Collision/NeighborPointQuery.h"
namespace dyno
{
	IMPLEMENT_TCLASS(VirtualParticleShiftingStrategy, TDataType)

		template <typename Real, typename Coord>
	__global__ void T_ComputeLambdas(
		DArray<Real> lambdaArr,
		DArray<Real> rhoArr,
		DArray<Coord> v_posArr,
		DArrayList<int> vv_neighbors,
		SpikyKernel<Real> kern,
		Real maxDensity,
		Real smoothingLength)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= v_posArr.size()) return;

		Coord pos_i = v_posArr[pId];

		Real lamda_i = Real(0);
		Coord grad_ci(0);

		List<int>& list_i = vv_neighbors[pId];
		int nbSize = list_i.size();
		for (int ne = 0; ne < nbSize; ne++)
		{

			int j = list_i[ne];
			if (j == pId) continue;
			Real r = (pos_i - v_posArr[j]).norm();

			if (r > EPSILON)
			{
				Coord g = kern.Gradient(r, smoothingLength) * (pos_i - v_posArr[j]) * (1.0f / r);
				grad_ci += g;
				lamda_i += g.dot(g);

			}
		}

		lamda_i += grad_ci.dot(grad_ci);

		Real rho_i = rhoArr[pId];
		
		lamda_i = -(rho_i - maxDensity) / (lamda_i + 0.1f);
		lambdaArr[pId] = lamda_i > 0.0f ? 0.0f : lamda_i;
	}

	template <typename Real, typename Coord>
	__global__ void T_ComputeDisplacement(
		DArray<Coord> dPos,
		DArray<Real> lambdas,
		DArray<Coord> v_posArr,
		DArrayList<int> vv_neighbors,
		SpikyKernel<Real> kern,
		Real smoothingLength,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= v_posArr.size()) return;
		

		Coord pos_i = v_posArr[pId];
		Real lamda_i = lambdas[pId];

		Coord dP_i(0);
		List<int>& list_i = vv_neighbors[pId];
		int nbSize = list_i.size();
		for (int ne = 0; ne < nbSize; ne++)
		{

			int j = list_i[ne];
			if (j == pId) continue;
			Real r = (pos_i - v_posArr[j]).norm();
			if (r > EPSILON)
			{
				Coord dp_ij = 20.0f * (pos_i - v_posArr[j]) * (lamda_i + lambdas[j]) * kern.Gradient(r, smoothingLength) * (1.0 / r);
				dP_i += dp_ij;

				atomicAdd(&dPos[pId][0], dp_ij[0]);

				if (Coord::dims() >= 2)
				{
					atomicAdd(&dPos[pId][1], dp_ij[1]);
				}

				if (Coord::dims() >= 3)
				{
					atomicAdd(&dPos[pId][2], dp_ij[2]);

				}
			}
		}

	}

	template <typename Real, typename Coord>
	__global__ void T_UpdatePosition(
		DArray<Coord> v_posArr,
		DArray<Coord> dPos,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= v_posArr.size()) return;
		v_posArr[pId] += dPos[pId];


	}



	template <typename Real, typename Coord>
	__global__ void T_RealCopytoVirtual(
		DArray<Coord> r_posArr,
		DArray<Coord> v_posArr,
		Real dt
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= r_posArr.size()) return;

		v_posArr[pId] = r_posArr[pId];

	}


	template<typename TDataType>
	VirtualParticleShiftingStrategy<TDataType>::VirtualParticleShiftingStrategy()
		: VirtualParticleGenerator<TDataType>()
	{
		this->varIterationNumber()->setValue(5);
		maxDensity = this->varRestDensity()->getValue();

		this->outVirtualParticles()->allocate();

		this->varSamplingDistance()->setValue(Real(0.005));
		this->varSmoothingLength()->setValue(Real(0.011));
		this->varRestDensity()->setValue(Real(1000));
		this->outVVNeighborIds()->allocate();
		
		///*Virtual particles' virtual neighbors*/
		m_vv_nbrQuery = std::make_shared<NeighborPointQuery<TDataType>>();
		this->varSmoothingLength()->connect(m_vv_nbrQuery->inRadius());
		this->outVirtualParticles()->connect(m_vv_nbrQuery->inPosition());
		m_vv_nbrQuery->outNeighborIds()->connect(this->outVVNeighborIds());

		m_vv_density = std::make_shared<SummationDensity<TDataType>>();
		this->varRestDensity()->connect(m_vv_density->varRestDensity());
		this->varSmoothingLength()->connect(m_vv_density->inSmoothingLength());
		this->varSamplingDistance()->connect(m_vv_density->inSamplingDistance());
		this->outVirtualParticles()->connect(m_vv_density->inPosition());
		this->outVVNeighborIds()->connect(m_vv_density->inNeighborIds());
		m_vv_density->outDensity()->connect(this->outVDensity());

	}

	template<typename TDataType>
	VirtualParticleShiftingStrategy<TDataType>::~VirtualParticleShiftingStrategy()
	{
		m_lamda.clear();
		m_deltaPos.clear();
		//	m_position_old.clear();
	}

	template<typename TDataType>
	bool VirtualParticleShiftingStrategy<TDataType>::VectorResize()
	{

		if (this->outVDensity()->size() != this->outVirtualParticles()->size())
			this->outVDensity()->resize(this->outVirtualParticles()->size());

		if (m_deltaPos.size() != this->inRPosition()->size())
			m_deltaPos.resize(this->inRPosition()->size());

		if (m_lamda.size() != this->inRPosition()->size())
			m_lamda.resize(this->inRPosition()->size());

		return true;
	}


	template<typename TDataType>
	void VirtualParticleShiftingStrategy<TDataType>::constrain()
	{

		VectorResize();

		int it = 0;

		int itNum = this->varIterationNumber()->getData();

		int num = this->inRPosition()->size();

		if (num != this->outVirtualParticles()->size())
		{
			this->outVirtualParticles()->resize(num);
		}

		cuExecute(num, T_RealCopytoVirtual,
			this->inRPosition()->getData(),
			this->outVirtualParticles()->getData(),
			this->inTimeStep()->getData()
		);

		m_vv_nbrQuery->compute();

		if (this->inFrameNumber()->getValue() == 0)
		{
			m_vv_density->update();
			Reduction<Real> reduce;
			maxDensity = reduce.maximum(m_vv_density->outDensity()->getData().begin(), m_vv_density->outDensity()->getData().size());
		}

		while (it < itNum)
		{
			takeOneIteration();
			it++;
		}
		std::cout << "*DUAL-ISPH::ParticleShiftingStrategy(S.B.)::Iteration:" << it << std::endl;
	}


	template<typename TDataType>
	void VirtualParticleShiftingStrategy<TDataType>::takeOneIteration()
	{

		Real dt = this->inTimeStep()->getData();
		int num = this->inRPosition()->size();
		uint pDims = cudaGridSize(num, BLOCK_SIZE);

		m_deltaPos.reset();

		m_vv_density->update();
		

		cuExecute(num, T_ComputeLambdas,
			m_lamda,
			m_vv_density->outDensity()->getData(),
			this->outVirtualParticles()->getData(),
			this->outVVNeighborIds()->getData(),
			m_kernel,
			maxDensity,
			this->varSmoothingLength()->getData());

		cuExecute(num, T_ComputeDisplacement,
			m_deltaPos,
			m_lamda,
			this->outVirtualParticles()->getData(),
			this->outVVNeighborIds()->getData(),
			m_kernel,
			this->varSmoothingLength()->getData(),
			dt);

		cuExecute(num, T_UpdatePosition,
			this->outVirtualParticles()->getData(),
			m_deltaPos,
			dt);
	}


	DEFINE_CLASS(VirtualParticleShiftingStrategy);

}