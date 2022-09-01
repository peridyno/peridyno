#include "DensityPBD.h"

#include "Node.h"
#include "SummationDensity.h"

namespace dyno
{
//	IMPLEMENT_TCLASS(DensityPBD, TDataType)

	template <typename Real, typename Coord>
	__global__ void K_ComputeLambdas(
		DArray<Real> lambdaArr,
		DArray<Real> rhoArr,
		DArray<Coord> posArr,
		DArrayList<int> neighbors,
		SpikyKernel<Real> kern,
		Real smoothingLength)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.size()) return;

		Coord pos_i = posArr[pId];

		Real lamda_i = Real(0);
		Coord grad_ci(0);

		List<int>& list_i = neighbors[pId];
		int nbSize = list_i.size();
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = list_i[ne];
			Real r = (pos_i - posArr[j]).norm();

			if (r > EPSILON)
			{
				Coord g = kern.Gradient(r, smoothingLength)*(pos_i - posArr[j]) * (1.0f / r);
				grad_ci += g;
				lamda_i += g.dot(g);
			}
		}

		lamda_i += grad_ci.dot(grad_ci);

		Real rho_i = rhoArr[pId];

		lamda_i = -(rho_i - 1000.0f) / (lamda_i + 0.1f);

		lambdaArr[pId] = lamda_i > 0.0f ? 0.0f : lamda_i;
	}

	template <typename Real, typename Coord>
	__global__ void K_ComputeDisplacement(
		DArray<Coord> dPos, 
		DArray<Real> lambdas, 
		DArray<Coord> posArr, 
		DArrayList<int> neighbors, 
		SpikyKernel<Real> kern,
		Real smoothingLength,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.size()) return;

		Coord pos_i = posArr[pId];
		Real lamda_i = lambdas[pId];

		Coord dP_i(0);
		List<int>& list_i = neighbors[pId];
		int nbSize = list_i.size();
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = list_i[ne];
			Real r = (pos_i - posArr[j]).norm();
			if (r > EPSILON)
			{
				Coord dp_ij = 10.0f*(pos_i - posArr[j])*(lamda_i + lambdas[j])*kern.Gradient(r, smoothingLength)* (1.0 / r);
				dP_i += dp_ij;
				
				atomicAdd(&dPos[pId][0], dp_ij[0]);
				atomicAdd(&dPos[j][0], -dp_ij[0]);

				if (Coord::dims() >= 2)
				{
					atomicAdd(&dPos[pId][1], dp_ij[1]);
					atomicAdd(&dPos[j][1], -dp_ij[1]);
				}
				
				if (Coord::dims() >= 3)
				{
					atomicAdd(&dPos[pId][2], dp_ij[2]);
					atomicAdd(&dPos[j][2], -dp_ij[2]);
				}
			}
		}

//		dPos[pId] = dP_i;
	}

	template <typename Real, typename Coord>
	__global__ void K_UpdatePosition(
		DArray<Coord> posArr, 
		DArray<Coord> velArr, 
		DArray<Coord> dPos, 
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.size()) return;

		posArr[pId] += dPos[pId];
	}


	template<typename TDataType>
	DensityPBD<TDataType>::DensityPBD()
		: ConstraintModule()
	{
		this->varIterationNumber()->setValue(3);

		this->varSamplingDistance()->setValue(Real(0.005));
		this->varSmoothingLength()->setValue(Real(0.011));
		this->varRestDensity()->setValue(Real(1000));

		m_summation = std::make_shared<SummationDensity<TDataType>>();

		this->varRestDensity()->connect(m_summation->varRestDensity());
		this->varSmoothingLength()->connect(m_summation->inSmoothingLength());
		this->varSamplingDistance()->connect(m_summation->inSamplingDistance());

		this->inPosition()->connect(m_summation->inPosition());
		this->inNeighborIds()->connect(m_summation->inNeighborIds());

		m_summation->outDensity()->connect(this->outDensity());
	}

	template<typename TDataType>
	DensityPBD<TDataType>::~DensityPBD()
	{
		m_lamda.clear();
		m_deltaPos.clear();
		m_position_old.clear();
	}

	template<typename TDataType>
	void DensityPBD<TDataType>::constrain()
	{
		int num = this->inPosition()->size();

		if (m_position_old.size() != this->inPosition()->size())
			m_position_old.resize(this->inPosition()->size());

		m_position_old.assign(this->inPosition()->getData());

		if (this->outDensity()->size() != this->inPosition()->size())
			this->outDensity()->resize(this->inPosition()->size());

		if (m_deltaPos.size() != this->inPosition()->size())
			m_deltaPos.resize(this->inPosition()->size());

		if (m_lamda.size() != this->inPosition()->size())
			m_lamda.resize(this->inPosition()->size());

		int it = 0;

		int itNum = this->varIterationNumber()->getData();
		while (it < itNum)
		{
			takeOneIteration();

			it++;
		}

		updateVelocity();
	}


	template<typename TDataType>
	void DensityPBD<TDataType>::takeOneIteration()
	{
		Real dt = this->inTimeStep()->getData();

		int num = this->inPosition()->size();
		uint pDims = cudaGridSize(num, BLOCK_SIZE);

		
		m_deltaPos.reset();

		m_summation->update();

		cuExecute(num, K_ComputeLambdas,
			m_lamda,
			m_summation->outDensity()->getData(),
			this->inPosition()->getData(),
			this->inNeighborIds()->getData(),
			m_kernel,
			this->varSmoothingLength()->getData());

		cuExecute(num, K_ComputeDisplacement,
			m_deltaPos,
			m_lamda,
			this->inPosition()->getData(),
			this->inNeighborIds()->getData(),
			m_kernel,
			this->varSmoothingLength()->getData(),
			dt);

		cuExecute(num, K_UpdatePosition,
			this->inPosition()->getData(),
			this->inVelocity()->getData(),
			m_deltaPos,
			dt);
	}

	template <typename Real, typename Coord>
	__global__ void DP_UpdateVelocity(
		DArray<Coord> velArr,
		DArray<Coord> prePos,
		DArray<Coord> curPos,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= velArr.size()) return;

		velArr[pId] += (curPos[pId] - prePos[pId]) / dt;

	}

	template<typename TDataType>
	void DensityPBD<TDataType>::updateVelocity()
	{
		int num = this->inPosition()->size();

		Real dt = this->inTimeStep()->getData();

		cuExecute(num, DP_UpdateVelocity,
			this->inVelocity()->getData(),
			m_position_old,
			this->inPosition()->getData(),
			dt);
	}

	DEFINE_CLASS(DensityPBD);
}