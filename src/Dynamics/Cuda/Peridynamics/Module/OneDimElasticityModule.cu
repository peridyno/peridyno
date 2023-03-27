#include <cuda_runtime.h>
#include "Node.h"
#include "OneDimElasticityModule.h"

namespace dyno
{
	template <typename Real, typename Coord>
	__global__ void ODE_SolveElasticityWithPBD(
		DArray<Coord> position_new,
		DArray<Coord> position,
		DArray<Real> mass,
		Real distance,
		Real lambda_prime)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;

		Coord delta_p = Coord(0);

		
		if (pId < position.size() - 1)
		{
			Coord p1 = position[pId];
			Coord p2 = position[pId + 1];

			Real w1 = Real(1) / mass[pId];
			Real w2 = Real(1) / mass[pId + 1];

			Coord d12 = p1 - p2;
			Real d12_norm = d12.norm();
			Coord n_12 = d12;
			if (d12_norm > EPSILON)
			{
				n_12.normalize();
			}
			else
			{
				n_12 = Coord(1, 0, 0);;
			}
			//Coord n_12 = d12_norm > EPSILON ? d12.normalize() : Coord(1, 0, 0);

			delta_p += -w1 / (w1 + w2)*(d12_norm - distance)*n_12;
			//Coord delta_p2 = w2 / (w1 + w2)*(d12_norm - distance)*n_12;
		}
		
		if (pId >= 1)
		{
			Coord p0 = position[pId - 1];
			Coord p1 = position[pId];

			Real w0 = Real(1) / mass[pId - 1];
			Real w1 = Real(1) / mass[pId];

			Coord d01 = p0 - p1;
			Real d01_norm = d01.norm();
			Coord n_01 = d01;
			if (d01_norm > EPSILON)
			{
				n_01.normalize();
			}
			else
			{
				n_01 = Coord(1, 0, 0);
			}
			//Coord n_01 = d01_norm > EPSILON ? d01.normalize() : Coord(1, 0, 0);

			//Coord delta_p0 = -w0 / (w0 + w1)*(d01_norm - distance)*n_01;
			delta_p += w1 / (w0 + w1)*(d01_norm - distance)*n_01;
		}

		position_new[pId] = position[pId] + lambda_prime *delta_p;
	}

	template <typename Real, typename Coord>
	__global__ void ODE_UpdateVelocity(
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
	OneDimElasticityModule<TDataType>::OneDimElasticityModule()
		: ConstraintModule()
	{
		m_distance.setValue(0.005);
 		m_lambda.setValue(0.1);
		m_iterNum.setValue(10);
	}


	template<typename TDataType>
	OneDimElasticityModule<TDataType>::~OneDimElasticityModule()
	{
	}

	template<typename TDataType>
	void OneDimElasticityModule<TDataType>::solveElasticity()
	{
		//Save new positions
		m_position_old.assign(m_position.getData());
		
		int itor = 0;
		Real lambda_prime = 1 - pow(1 - m_lambda.getData(), 1 / Real(m_iterNum.getData()));
		while (itor < m_iterNum.getData())
		{
			m_position_buf.assign(m_position.getData());

			int num = m_position.size();
			uint pDims = cudaGridSize(num, BLOCK_SIZE);

			ODE_SolveElasticityWithPBD << <pDims, BLOCK_SIZE >> > (
				m_position.getData(),
				m_position_buf,
				m_mass.getData(),
				m_distance.getData(),
				lambda_prime);

			itor++;
		}

		this->updateVelocity();
	}

	template<typename TDataType>
	void OneDimElasticityModule<TDataType>::updateVelocity()
	{
		int num = m_position.size();
		uint pDims = cudaGridSize(num, BLOCK_SIZE);

		Real dt = 0.001;

		ODE_UpdateVelocity << <pDims, BLOCK_SIZE >> > (
			m_velocity.getData(),
			m_position_old,
			m_position.getData(),
			dt);
		cuSynchronize();
	}


	template<typename TDataType>
	void OneDimElasticityModule<TDataType>::constrain()
	{
		this->solveElasticity();
	}


	template<typename TDataType>
	bool OneDimElasticityModule<TDataType>::initializeImpl()
	{
		if (m_distance.isEmpty() || m_position.isEmpty() || m_velocity.isEmpty())
		{
			std::cout << "Exception: " << std::string("ElasticityModule's fields are not fully initialized!") << "\n";
			return false;
		}

		int num = m_position.size();
		
// 		m_invK.resize(num);
// 		m_weights.resize(num);
// 		m_displacement.resize(num);
// 
// 		m_F.resize(num);
// 		
 		m_position_old.resize(num);
		m_position_buf.resize(num);
// 		m_bulkCoefs.resize(num);
// 
// 		resetRestShape();
// 
// 		this->computeMaterialStiffness();
// 
// 		Function1Pt::copy(m_position_old, m_position.getData());

		return true;
	}

	DEFINE_CLASS(OneDimElasticityModule);
}